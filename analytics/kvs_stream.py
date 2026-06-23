import os
import sys
import logging
import platform
import subprocess
import boto3
from botocore.exceptions import ClientError, NoCredentialsError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

AWS_REGION   = os.getenv("AWS_REGION", "eu-central-1")
STREAM_NAME  = os.getenv("KVS_STREAM_NAME", "dispenser-live-feed")
PROFILE_NAME = os.getenv("AWS_PROFILE", "dispenser_video")

IS_RASPBERRY_PI = platform.machine() in ("aarch64", "armv7l")
IS_MAC          = platform.system() == "Darwin"


class KVSStreamError(RuntimeError):
    """Recoverable / configuration errors when embedding (no sys.exit)."""


def get_kvs_client():
    """Returns the Boto3 Kinesis Video client."""
    try:
        session = boto3.Session(profile_name=PROFILE_NAME, region_name=AWS_REGION)
        client = session.client("kinesisvideo")
        client.describe_stream(StreamName=STREAM_NAME)
        log.info("KVS stream '%s' bulundu.", STREAM_NAME)
        return client
    except NoCredentialsError as e:
        log.error("AWS credentials not found. Run 'aws configure --profile %s'.", PROFILE_NAME)
        raise KVSStreamError("No AWS credentials") from e
    except ClientError as e:
        log.error("AWS hatası: %s", e)
        raise KVSStreamError(str(e)) from e


def get_data_endpoint(kvs_client) -> str:
    """Retrieves the data endpoint URL for PUT requests to the stream."""
    resp = kvs_client.get_data_endpoint(StreamName=STREAM_NAME, APIName="PUT_MEDIA")
    return resp["DataEndpoint"]


def build_gst_pipeline(endpoint: str) -> str:
    """Returns the GStreamer pipeline string for the platform."""
    kvs_sink = (
        f"kvssink stream-name={STREAM_NAME} "
        f"storage-size=128 "
        f"aws-region={AWS_REGION}"
    )

    encode = "videoconvert ! x264enc bframes=0 key-int-max=45 bitrate=512 ! video/x-h264,stream-format=avc,alignment=au,profile=baseline ! "

    if IS_RASPBERRY_PI:
        log.info("Platform: Raspberry Pi → libcamerasrc.")
        pipeline = (
            f"gst-launch-1.0 libcamerasrc ! "
            f"video/x-raw,width=640,height=480,framerate=15/1,format=RGB ! "
            f"{encode}{kvs_sink}"
        )
    elif IS_MAC:
        log.info("Platform: macOS → avfvideosrc.")
        pipeline = (
            f"gst-launch-1.0 avfvideosrc device-index=0 ! "
            f"video/x-raw,width=640,height=480,framerate=15/1 ! "
            f"{encode}{kvs_sink}"
        )
    else:
        log.info("Platform: other → videotestsrc.")
        pipeline = (
            f"gst-launch-1.0 videotestsrc ! "
            f"video/x-raw,width=640,height=480,framerate=15/1 ! "
            f"{encode}{kvs_sink}"
        )

    return pipeline


def check_gstreamer() -> bool:
    """Checks if GStreamer is installed."""
    result = subprocess.run(["gst-launch-1.0", "--version"], capture_output=True, text=True)
    return result.returncode == 0


def stream_to_kinesis():
    """
    Run one GStreamer → KVS session until the process exits or fails.
    When embedded (e.g. Pi API thread), raises KVSStreamError instead of sys.exit.
    """
    log.info("=== Starting Drug Dispenser KVS Stream ===")
    log.info("Region: %s | Stream: %s", AWS_REGION, STREAM_NAME)

    if not check_gstreamer():
        log.error("GStreamer not found.")
        raise KVSStreamError("GStreamer not available")

    kvs = get_kvs_client()
    endpoint = get_data_endpoint(kvs)
    log.info("Data endpoint: %s", endpoint)

    pipeline_cmd = build_gst_pipeline(endpoint)
    log.info("GStreamer pipeline:\n  %s", pipeline_cmd)

    log.info("Stream started. Press Ctrl+C to stop.")
    try:
        subprocess.run(pipeline_cmd.split(), check=True)
    except KeyboardInterrupt:
        log.info("Stream stopped.")
    except subprocess.CalledProcessError as e:
        log.error("GStreamer error: %s", e)
        raise KVSStreamError(f"gst-launch failed: {e}") from e


def stream_to_kinesis_forever(retry_delay_sec: float = 30.0) -> None:
    """
    Keep trying to stream to KVS until the process exits. For use as a daemon
    thread alongside the Flask API (Pi açık olduğu sürece yeniden dener).
    """
    import time as _time

    while True:
        try:
            stream_to_kinesis()
            log.warning("KVS stream ended normally; restarting in %ss", retry_delay_sec)
        except KVSStreamError as e:
            log.warning("KVS stream error (retry in %ss): %s", retry_delay_sec, e)
        except Exception as e:
            log.exception("KVS unexpected error (retry in %ss): %s", retry_delay_sec, e)
        _time.sleep(retry_delay_sec)


if __name__ == "__main__":
    try:
        stream_to_kinesis()
    except KVSStreamError as e:
        log.error("%s", e)
        sys.exit(1)