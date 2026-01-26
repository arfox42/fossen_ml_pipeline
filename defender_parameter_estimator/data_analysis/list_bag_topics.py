#!/usr/bin/env python3
"""
List all topics and message types from a ROS 2 (Iron+) MCAP bag.
"""

import sys
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions

def list_topics(bag_path: str):
    # Use 'mcap' since your bag has .mcap files
    storage_options = StorageOptions(uri=bag_path, storage_id='mcap')
    converter_options = ConverterOptions(input_serialization_format='cdr',
                                         output_serialization_format='cdr')

    reader = SequentialReader()
    reader.open(storage_options, converter_options)

    topics = reader.get_all_topics_and_types()

    print(f"\n=== Topics in bag: {bag_path} ===")
    for t in topics:
        print(f"{t.name:<45} | type: {t.type}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 list_bag_topics.py <path_to_ros2_bag>")
        sys.exit(1)

    list_topics(sys.argv[1])
