#!/usr/bin/env python3

import argparse
import base64
import json
import time
import urllib.error
import urllib.request

import cv2
import numpy as np


def decode_jpeg_base64(encoded, field_name):
    data = base64.b64decode(encoded)
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to decode {field_name}")
    return image


def fetch_images(url, timeout_sec):
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout_sec) as response:
        body = json.loads(response.read().decode("utf-8"))

    if not body.get("success", False):
        raise RuntimeError(body.get("message", "request failed"))

    debug_image = decode_jpeg_base64(
        body["debug_image_jpeg_base64"],
        "debug_image_jpeg_base64")
    object_mask_image = decode_jpeg_base64(
        body["object_mask_image_jpeg_base64"],
        "object_mask_image_jpeg_base64")
    return debug_image, object_mask_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="192.168.2.107")
    parser.add_argument("--port", type=int, default=13604)
    parser.add_argument("--interval-ms", type=int, default=200)
    parser.add_argument("--timeout-sec", type=float, default=5.0)
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}/foundationpose/visualization_images"
    interval_sec = max(args.interval_ms, 1) / 1000.0

    while True:
        try:
            debug_image, object_mask_image = fetch_images(url, args.timeout_sec)
            cv2.imshow("FoundationPose Debug", debug_image)
            cv2.imshow("FoundationPose Object Mask", object_mask_image)
            key = cv2.waitKey(1) & 0xff
            if key in (ord("q"), 27):
                break
        except urllib.error.HTTPError as e:
            message = e.read().decode("utf-8", errors="replace")
            try:
                message = json.loads(message).get("message", message)
            except json.JSONDecodeError:
                pass
            print(f"HTTP {e.code}: {message}")
        except Exception as e:
            print(e)

        time.sleep(interval_sec)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
