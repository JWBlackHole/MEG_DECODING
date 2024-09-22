import torch

if __name__=="__main__":
        
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} is available")
    else:
        print("no GPU!")