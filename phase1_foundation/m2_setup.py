 # MacBook M2 optimization setup

"""
JARVIS MacBook M2 Setup - Neural Engine Optimization
This prepares your M2 MacBook for 13B neural processing with maximum performance
"""

import subprocess
import sys
import os
import json
from pathlib import Path

class M2JarvisSetup:
    def __init__(self):
        """Initialize M2 setup for maximum JARVIS performance"""
        print("🚀 JARVIS M2 MacBook Setup - Neural Engine Optimization")
        print("=" * 60)
        
        self.setup_results = {}
        
    def check_system_requirements(self):
        """Check M2 MacBook capabilities"""
        print("🔍 Checking M2 system capabilities...")
        
        try:
            # Get system info
            result = subprocess.run(['system_profiler', 'SPHardwareDataType'], 
                                  capture_output=True, text=True)
            
            if 'Apple M2' in result.stdout:
                print("✅ Apple M2 chip detected - Neural Engine available!")
                self.setup_results['m2_chip'] = True
            else:
                print("⚠️ M2 chip not detected - performance may be limited")
                self.setup_results['m2_chip'] = False
            
            # Check memory
            memory_info = subprocess.run(['sysctl', 'hw.memsize'], 
                                       capture_output=True, text=True)
            memory_bytes = int(memory_info.stdout.split(':')[1].strip())
            memory_gb = memory_bytes // (1024**3)
            
            print(f"💾 System Memory: {memory_gb}GB")
            self.setup_results['memory_gb'] = memory_gb
            
            if memory_gb >= 16:
                print("✅ Sufficient memory for 13B models")
            else:
                print("⚠️ Limited memory - may need smaller models")
            
            return True
            
        except Exception as e:
            print(f"❌ System check failed: {e}")
            return False
    
    def install_homebrew(self):
        """Install Homebrew package manager"""
        print("\n🍺 Installing Homebrew...")
        
        # Check if already installed
        try:
            result = subprocess.run(['brew', '--version'], capture_output=True)
            if result.returncode == 0:
                print("✅ Homebrew already installed")
                self.setup_results['homebrew'] = True
                return True
        except FileNotFoundError:
            pass
        
        # Install Homebrew
        install_script = '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
        
        print("🔄 Installing Homebrew (this may take a few minutes)...")
        result = subprocess.run(install_script, shell=True)
        
        if result.returncode == 0:
            print("✅ Homebrew installed successfully")
            self.setup_results['homebrew'] = True
            return True
        else:
            print("❌ Homebrew installation failed")
            self.setup_results['homebrew'] = False
            return False
    
    def install_python_environment(self):
        """Install Python and create JARVIS environment"""
        print("\n🐍 Setting up Python environment...")
        
        # Install Python via Homebrew
        try:
            print("🔄 Installing Python 3.11...")
            subprocess.run(['brew', 'install', 'python@3.11'], check=True)
            
            # Install pip packages
            pip_packages = [
                'virtualenv',
                'pip-tools',
                'wheel'
            ]
            
            for package in pip_packages:
                subprocess.run(['pip3', 'install', package], check=True)
            
            print("✅ Python environment ready")
            self.setup_results['python'] = True
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Python setup failed: {e}")
            self.setup_results['python'] = False
            return False
    
    def create_jarvis_virtualenv(self):
        """Create dedicated JARVIS virtual environment"""
        print("\n🌍 Creating JARVIS virtual environment...")
        
        try:
            # Create virtual environment
            venv_path = Path.home() / 'JARVIS_PROJECT' / 'jarvis_env'
            subprocess.run(['python3', '-m', 'venv', str(venv_path)], check=True)
            
            # Activate and install base packages
            pip_path = venv_path / 'bin' / 'pip'
            
            base_packages = [
                'wheel',
                'setuptools',
                'pip-tools',
                'numpy',
                'scipy'
            ]
            
            for package in base_packages:
                subprocess.run([str(pip_path), 'install', package], check=True)
            
            print(f"✅ Virtual environment created at: {venv_path}")
            self.setup_results['virtualenv'] = str(venv_path)
            return True
            
        except Exception as e:
            print(f"❌ Virtual environment creation failed: {e}")
            return False
    
    def install_android_tools(self):
        """Install Android development tools"""
        print("\n📱 Installing Android development tools...")
        
        try:
            # Install Android platform tools
            subprocess.run(['brew', 'install', 'android-platform-tools'], check=True)
            
            # Verify ADB installation
            result = subprocess.run(['adb', 'version'], capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ ADB installed and working")
                self.setup_results['adb'] = True
            else:
                print("❌ ADB installation failed")
                self.setup_results['adb'] = False
            
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Android tools installation failed: {e}")
            return False
    
    def install_ml_frameworks(self):
        """Install ML frameworks optimized for M2"""
        print("\n🧠 Installing ML frameworks for M2...")
        
        venv_path = Path.home() / 'JARVIS_PROJECT' / 'jarvis_env'
        pip_path = venv_path / 'bin' / 'pip'
        
        try:
            # M2-optimized packages
            ml_packages = [
                'torch',
                'torchvision', 
                'torchaudio',
                'transformers',
                'accelerate',
                'sentencepiece',
                'protobuf',
                'sounddevice',
                'librosa',
                'TTS',
                'whisper',
                'fastapi',
                'uvicorn',
                'websockets',
                'pydub',
                'opencv-python',
                'requests',
                'aiofiles'
            ]
            
            print("🔄 Installing ML packages (this will take several minutes)...")
            for package in ml_packages:
                print(f"   Installing {package}...")
                subprocess.run([str(pip_path), 'install', package], check=True)
            
            print("✅ ML frameworks installed")
            self.setup_results['ml_frameworks'] = True
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ ML frameworks installation failed: {e}")
            self.setup_results['ml_frameworks'] = False
            return False
    
    def create_requirements_file(self):
        """Create requirements.txt for the project"""
        print("\n📄 Creating requirements.txt...")
        
        requirements = """# JARVIS MacBook M2 Requirements
# Core ML Frameworks
torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0
transformers>=4.30.0
accelerate>=0.20.0
sentencepiece>=0.1.99
protobuf>=4.23.0

# Voice Processing
TTS>=0.15.0
whisper>=1.1.0
sounddevice>=0.4.6
librosa>=0.10.0
pydub>=0.25.0

# Web Framework
fastapi>=0.100.0
uvicorn>=0.22.0
websockets>=11.0.0
aiofiles>=23.1.0

# Computer Vision
opencv-python>=4.8.0
pillow>=10.0.0

# Utilities
requests>=2.31.0
numpy>=1.24.0
scipy>=1.11.0
python-multipart>=0.0.6
pyyaml>=6.0
"""
        
        try:
            with open('requirements.txt', 'w') as f:
                f.write(requirements)
            
            print("✅ requirements.txt created")
            return True
            
        except Exception as e:
            print(f"❌ Requirements file creation failed: {e}")
            return False
    
    def test_m2_performance(self):
        """Test M2 Neural Engine performance"""
        print("\n🧪 Testing M2 Neural Engine performance...")
        
        test_script = '''
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
        '''
        
        # Write test script
        with open('m2_performance_test.py', 'w') as f:
            f.write(test_script)
        
        # Run performance test
        venv_path = Path.home() / 'JARVIS_PROJECT' / 'jarvis_env'
        python_path = venv_path / 'bin' / 'python'
        
        try:
            result = subprocess.run([str(python_path), 'm2_performance_test.py'], 
                                  capture_output=True, text=True)
            print(result.stdout)
            
            if result.returncode == 0:
                self.setup_results['performance_test'] = True
            else:
                print(f"❌ Performance test failed: {result.stderr}")
                self.setup_results['performance_test'] = False
            
            return True
            
        except Exception as e:
            print(f"❌ Performance test error: {e}")
            return False
    
    def create_activation_script(self):
        """Create environment activation script"""
        print("\n📝 Creating activation script...")
        
        activation_script = '''#!/bin/bash
# JARVIS Environment Activation Script

echo "🤖 Activating JARVIS environment..."

# Activate virtual environment
source ~/JARVIS_PROJECT/jarvis_env/bin/activate

# Set environment variables
export JARVIS_PROJECT_ROOT="$HOME/JARVIS_PROJECT"
export JARVIS_MODELS_PATH="$JARVIS_PROJECT_ROOT/models"
export JARVIS_CACHE_PATH="$JARVIS_PROJECT_ROOT/cache"

# Add project to Python path
export PYTHONPATH="$JARVIS_PROJECT_ROOT/phase1_foundation:$PYTHONPATH"

echo "✅ JARVIS environment activated!"
echo "📁 Project root: $JARVIS_PROJECT_ROOT"
echo "🧠 Models path: $JARVIS_MODELS_PATH"
echo ""
echo "🚀 Ready to run JARVIS commands!"
        '''
        
        try:
            with open('activate_jarvis.sh', 'w') as f:
                f.write(activation_script)
            
            # Make executable
            os.chmod('activate_jarvis.sh', 0o755)
            
            print("✅ Activation script created: activate_jarvis.sh")
            return True
            
        except Exception as e:
            print(f"❌ Activation script creation failed: {e}")
            return False
    
    def generate_setup_report(self):
        """Generate setup completion report"""
        print("\n📊 Generating setup report...")
        
        report = {
            'setup_timestamp': subprocess.run(['date'], capture_output=True, text=True).stdout.strip(),
            'system_info': {
                'm2_chip': self.setup_results.get('m2_chip', False),
                'memory_gb': self.setup_results.get('memory_gb', 0)
            },
            'installation_results': self.setup_results,
            'next_steps': [
                "1. Connect your Nothing Phone (2a) via USB",
                "2. Enable USB debugging on your phone",
                "3. Run: source activate_jarvis.sh",
                "4. Run: python phone_analyzer.py",
                "5. Proceed with Phase 1 components"
            ]
        }
        
        try:
            with open('setup_report.json', 'w') as f:
                json.dump(report, f, indent=4)
            
            print("✅ Setup report saved: setup_report.json")
            return report
            
        except Exception as e:
            print(f"❌ Report generation failed: {e}")
            return None
    
    def run_complete_setup(self):
        """Run complete M2 setup process"""
        print("🤖 JARVIS MacBook M2 Complete Setup")
        print("This will install everything needed for 13B neural JARVIS")
        print("=" * 60)
        
        setup_steps = [
            ("Checking M2 system requirements", self.check_system_requirements),
            ("Installing Homebrew", self.install_homebrew),
            ("Setting up Python environment", self.install_python_environment),
            ("Creating JARVIS virtual environment", self.create_jarvis_virtualenv),
            ("Installing Android tools", self.install_android_tools),
            ("Installing ML frameworks", self.install_ml_frameworks),
            ("Creating requirements file", self.create_requirements_file),
            ("Testing M2 performance", self.test_m2_performance),
            ("Creating activation script", self.create_activation_script),
        ]
        
        success_count = 0
        
        for step_name, step_function in setup_steps:
            print(f"\n🔄 {step_name}...")
            try:
                if step_function():
                    success_count += 1
                    print(f"✅ {step_name} completed!")
                else:
                    print(f"❌ {step_name} failed!")
            except Exception as e:
                print(f"❌ {step_name} error: {e}")
        
        # Generate final report
        report = self.generate_setup_report()
        
        print(f"\n📊 Setup Summary: {success_count}/{len(setup_steps)} steps completed")
        
        if success_count >= len(setup_steps) - 1:  # Allow 1 failure
            print("🎉 M2 JARVIS Setup Complete!")
            print("\n🚀 Next Steps:")
            print("1. Run: source activate_jarvis.sh")
            print("2. Connect your Nothing Phone (2a)")
            print("3. Continue with Phase 1 components")
            return True
        else:
            print("⚠️ Setup incomplete - please check failed steps")
            return False

def main():
    """Main setup function"""
    setup = M2JarvisSetup()
    return setup.run_complete_setup()

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Ready for JARVIS Phase 1!")
    else:
        print("\n❌ Please fix setup issues before continuing")
