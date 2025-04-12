#!/usr/bin/env python3
from roop.core import face_swap_main
import argparse

def main():
    # Tạo đối tượng parser cho dòng lệnh
    parser = argparse.ArgumentParser(description="Faceswap with Roop")
    parser.add_argument('--source', required=True, help="Đường dẫn tới ảnh nguồn")
    parser.add_argument('--target', required=True, help="Đường dẫn tới ảnh đích")
    parser.add_argument('--output', required=True, help="Đường dẫn lưu ảnh đầu ra")
    
    args = parser.parse_args()
    
    # Gọi hàm faceswap từ roop.core
    face_swap_main(args.source, args.target, args.output)

if __name__ == '__main__':
    main()
