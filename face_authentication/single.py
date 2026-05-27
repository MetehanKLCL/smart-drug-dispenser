"""
Only 1 camera is opened at a time: register and auth run without closing the camera between them.
1 = Face register
2 = Authentication  
q = Quit
"""
import auth
import register
from camera import get_camera, release_camera


def main():
    cap = get_camera()
    if cap is None:
        print("Camera cannot be opened.")
        return
    try:
        while True:
            print("\n1 = Face register   2 = Authentication   q = Quit")
            choice = input("> ").strip().lower()
            if choice == "q":
                break
            if choice == "1":
                register.main(cap)
            elif choice == "2":
                auth.main(cap)
            else:
                print("Invalid choice.")
    finally:
        release_camera(cap)


if __name__ == "__main__":
    main()
