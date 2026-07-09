#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import rospy
from std_msgs.msg import String


def callback(msg):
    try:
        data = json.loads(msg.data)
    except Exception as e:
        rospy.logerr("JSON解析失败: %s", e)
        rospy.logerr("原始数据: %s", msg.data)
        return

    print("\n================ YOLO Boxes3D ================")
    print(f'frame_id: {data.get("frame_id")}')
    print(f'stamp:    {data.get("stamp")}')

    boxes = data.get("boxes", [])
    print(f"box_count: {len(boxes)}")

    for i, box in enumerate(boxes):
        print(f"\nbox[{i}]")
        print(f"  center   : {box['center']}")
        print(f"  quat     : {box['quat']}")
        print(f"  size     : {box['size']}")
        print(f"  score    : {box['score']:.3f}")
        print(f"  class_id : {box['class_id']}")

    print("==============================================")


if __name__ == "__main__":
    rospy.init_node("yolo_boxes_printer")
    rospy.Subscriber(
        "/yolo/target_boxes3d_string",
        String,
        callback,
        queue_size=1,
    )
    rospy.spin()