import torch

def check_devices():
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Device count:", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}:", torch.cuda.get_device_name(i))
    print("MPS available:", torch.backends.mps.is_available())

if __name__ == "__main__":
    check_devices()
