
import time
import torch
import numpy as np

def test_m2_performance():
    """Test M2 performance capabilities"""
    print("🔬 Testing M2 Neural Engine...")
    
    # Test MPS (Metal Performance Shaders) availability
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ MPS (Metal Performance Shaders) available")
    else:
        device = torch.device("cpu")
        print("⚠️ MPS not available, using CPU")
    
    # Performance test
    size = 2048
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    
    # Warm up
    torch.matmul(a, b)
    
    # Benchmark
    start_time = time.time()
    iterations = 10
    
    for _ in range(iterations):
        result = torch.matmul(a, b)
        if device.type == "mps":
            torch.mps.synchronize()
    
    end_time = time.time()
    avg_time = (end_time - start_time) / iterations
    
    # Calculate performance metrics
    operations = size * size * size * 2  # Matrix multiply operations
    gflops = operations / (avg_time * 1e9)
    
    print(f"📊 Performance Results:")
    print(f"   Device: {device}")
    print(f"   Matrix size: {size}x{size}")
    print(f"   Average time: {avg_time*1000:.2f}ms")
    print(f"   Performance: {gflops:.2f} GFLOPS")
    
    if gflops > 100:
        print("✅ Excellent performance for 13B models!")
    elif gflops > 50:
        print("✅ Good performance for AI processing")
    else:
        print("⚠️ Limited performance - consider optimizations")
    
    return gflops

if __name__ == "__main__":
    test_m2_performance()
        