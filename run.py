!/usr/bin/env python3
from roop.core import face_swap_main
import argparse
import torch

def main():
    # Kiểm tra GPU và thiết lập device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Tạo đối tượng parser cho dòng lệnh
    parser = argparse.ArgumentParser(description="Faceswap with Roop")
    parser.add_argument('--source', required=True, help="Đường dẫn tới ảnh nguồn")
    parser.add_argument('--target', required=True, help="Đường dẫn tới ảnh đích")
    parser.add_argument('--output', required=True, help="Đường dẫn lưu ảnh đầu ra")
    
    args = parser.parse_args()
    
    # Gọi hàm faceswap từ roop.core và chuyển mô hình/dữ liệu sang GPU nếu có
    face_swap_main(args.source, args.target, args.output, device=device)

if __name__ == '__main__':
    main()
