import os
import boto3
from botocore.exceptions import ClientError
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/stream", tags=["stream"])

KVS_REGION = os.getenv("KVS_REGION", "eu-central-1")
KVS_STREAM_NAME = os.getenv("KVS_STREAM_NAME", "dispenser-live-feed")


def _get_hls_url() -> str:
    """Retrieves the HLS streaming session URL from Kinesis."""
    kvs = boto3.client(
        "kinesisvideo",
        region_name=KVS_REGION,
        aws_access_key_id=os.getenv("KVS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("KVS_SECRET_ACCESS_KEY"),
    )

    # 1) Retrieve the HLS endpoint URL
    endpoint_resp = kvs.get_data_endpoint(
        StreamName=KVS_STREAM_NAME,
        APIName="GET_HLS_STREAMING_SESSION_URL"
    )
    endpoint_url = endpoint_resp["DataEndpoint"]

    # 2) Retrieve the HLS session URL (valid for 5 minutes)
    kvs_media = boto3.client(
        "kinesis-video-archived-media",
        endpoint_url=endpoint_url,
        region_name=KVS_REGION,
        aws_access_key_id=os.getenv("KVS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("KVS_SECRET_ACCESS_KEY"),
    )
    hls_resp = kvs_media.get_hls_streaming_session_url(
        StreamName=KVS_STREAM_NAME,
        PlaybackMode="LIVE",
        HLSFragmentSelector={
            "FragmentSelectorType": "SERVER_TIMESTAMP"
        },
        ContainerFormat="FRAGMENTED_MP4",
        DiscontinuityMode="ALWAYS",
        DisplayFragmentTimestamp="ALWAYS",
        Expires=300   # 5 minutes
    )
    return hls_resp["HLSStreamingSessionURL"]


@router.get("/live")
def get_live_stream_url():
    """
    Caregiver Flutter app calls this endpoint.
    Returns the HLS URL, Flutter video player plays it.

    GET /stream/live
    Response: { "hls_url": "https://..." }
    """
    try:
        hls_url = _get_hls_url()
        return {"hls_url": hls_url}
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code == "ResourceNotFoundException":
            raise HTTPException(
                status_code=404,
                detail=f"Stream not found: {KVS_STREAM_NAME}"
            )
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))