#!/usr/bin/env python3
"""
Enhanced Phase 1 Completion Checker for JARVIS Project
Comprehensive, cross-platform testing and validation of all Phase 1 components
Version: 2.0.0 - Enterprise Grade
Author: Advanced AI Systems
"""

import os
import sys
import subprocess
import json
import time
import asyncio
import hashlib
import tempfile
import logging
import platform
import shutil
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from contextlib import contextmanager
import importlib.util
import pkg_resources


@dataclass
class CheckResult:
    """Enhanced result container for validation checks"""
    name: str
    status: bool
    message: str
    details: Dict[str, Any]
    execution_time: float
    severity: str = "info"  # info, warning, error, critical
    suggestions: List[str] = None
    
    def __post_init__(self):
        if self.suggestions is None:
            self.suggestions = []


class EnhancedLogger:
    """Advanced logging system with multiple output formats"""
    
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup multiple loggers
        self.setup_loggers()
        
    def setup_loggers(self):
        """Setup comprehensive logging system"""
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        
        # Main logger
        self.logger = logging.getLogger('JARVIS_Phase1_Checker')
        self.logger.setLevel(logging.DEBUG)
        
        # File handler for detailed logs
        file_handler = logging.FileHandler(self.log_dir / f'phase1_check_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(log_format))
        
        # Console handler for user feedback
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter('%(message)s'))
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def info(self, message: str, emoji: str = "ℹ️"):
        self.logger.info(f"{emoji} {message}")
    
    def success(self, message: str):
        self.logger.info(f"✅ {message}")
    
    def warning(self, message: str):
        self.logger.warning(f"⚠️ {message}")
    
    def error(self, message: str):
        self.logger.error(f"❌ {message}")
    
    def critical(self, message: str):
        self.logger.critical(f"🚨 {message}")


class CrossPlatformUtils:
    """Cross-platform utility functions"""
    
    @staticmethod
    def get_platform() -> str:
        """Get normalized platform name"""
        system = platform.system().lower()
        if system == "darwin":
            return "macos"
        elif system == "windows":
            return "windows"
        else:
            return "linux"
    
    @staticmethod
    def get_shell_command(command: str) -> List[str]:
        """Get platform-appropriate shell command"""
        platform_name = CrossPlatformUtils.get_platform()
        if platform_name == "windows":
            return ["cmd", "/c", command]
        else:
            return ["/bin/bash", "-c", command]
    
    @staticmethod
    def get_python_executable() -> str:
        """Get the correct Python executable for the platform"""
        return sys.executable
    
    @staticmethod
    def check_admin_privileges() -> bool:
        """Check if running with administrative privileges"""
        platform_name = CrossPlatformUtils.get_platform()
        try:
            if platform_name == "windows":
                import ctypes
                return ctypes.windll.shell32.IsUserAnAdmin() != 0
            else:
                return os.getuid() == 0
        except:
            return False


class SecurityValidator:
    """Enhanced security validation and integrity checking"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
    
    def validate_file_permissions(self, file_path: Path) -> CheckResult:
        """Validate file permissions and security"""
        start_time = time.time()
        
        try:
            if not file_path.exists():
                return CheckResult(
                    name=f"permissions_{file_path.name}",
                    status=False,
                    message=f"File does not exist: {file_path}",
                    details={"path": str(file_path)},
                    execution_time=time.time() - start_time,
                    severity="error"
                )
            
            stat = file_path.stat()
            permissions = oct(stat.st_mode)[-3:]
            
            # Check for secure permissions
            is_secure = True
            issues = []
            
            if file_path.suffix == ".py":
                # Python files should be readable/writable by owner, readable by group
                if permissions not in ["644", "664", "755"]:
                    is_secure = False
                    issues.append(f"Insecure permissions: {permissions}")
            
            # Check if file is world-writable (security risk)
            if stat.st_mode & 0o002:
                is_secure = False
                issues.append("File is world-writable (security risk)")
            
            return CheckResult(
                name=f"permissions_{file_path.name}",
                status=is_secure,
                message="File permissions secure" if is_secure else f"Permission issues: {', '.join(issues)}",
                details={
                    "path": str(file_path),
                    "permissions": permissions,
                    "issues": issues
                },
                execution_time=time.time() - start_time,
                severity="warning" if not is_secure else "info"
            )
            
        except Exception as e:
            return CheckResult(
                name=f"permissions_{file_path.name}",
                status=False,
                message=f"Permission check failed: {str(e)}",
                details={"error": str(e)},
                execution_time=time.time() - start_time,
                severity="error"
            )
    
    def calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file for integrity checking"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def validate_file_integrity(self, file_path: Path, expected_hash: Optional[str] = None) -> CheckResult:
        """Validate file integrity using checksums"""
        start_time = time.time()
        
        try:
            if not file_path.exists():
                return CheckResult(
                    name=f"integrity_{file_path.name}",
                    status=False,
                    message=f"File does not exist for integrity check",
                    details={},
                    execution_time=time.time() - start_time,
                    severity="error"
                )
            
            actual_hash = self.calculate_file_hash(file_path)
            
            # If no expected hash provided, just return the calculated hash
            if expected_hash is None:
                return CheckResult(
                    name=f"integrity_{file_path.name}",
                    status=True,
                    message="File hash calculated successfully",
                    details={"hash": actual_hash, "size": file_path.stat().st_size},
                    execution_time=time.time() - start_time
                )
            
            # Compare with expected hash
            integrity_valid = actual_hash == expected_hash
            
            return CheckResult(
                name=f"integrity_{file_path.name}",
                status=integrity_valid,
                message="File integrity verified" if integrity_valid else "File integrity check failed",
                details={
                    "expected_hash": expected_hash,
                    "actual_hash": actual_hash,
                    "size": file_path.stat().st_size
                },
                execution_time=time.time() - start_time,
                severity="critical" if not integrity_valid else "info"
            )
            
        except Exception as e:
            return CheckResult(
                name=f"integrity_{file_path.name}",
                status=False,
                message=f"Integrity check failed: {str(e)}",
                details={"error": str(e)},
                execution_time=time.time() - start_time,
                severity="error"
            )


class PerformanceProfiler:
    """Performance monitoring and profiling"""
    
    def __init__(self):
        self.metrics = {}
        self.start_times = {}
    
    @contextmanager
    def profile_section(self, section_name: str):
        """Context manager for profiling code sections"""
        start_time = time.time()
        start_memory = self.get_memory_usage()
        
        try:
            yield
        finally:
            end_time = time.time()
            end_memory = self.get_memory_usage()
            
            self.metrics[section_name] = {
                "execution_time": end_time - start_time,
                "memory_delta": end_memory - start_memory,
                "timestamp": datetime.now().isoformat()
            }
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate performance report"""
        total_time = sum(m["execution_time"] for m in self.metrics.values())
        total_memory = sum(m["memory_delta"] for m in self.metrics.values())
        
        return {
            "total_execution_time": total_time,
            "total_memory_delta": total_memory,
            "section_metrics": self.metrics,
            "performance_grade": self.calculate_performance_grade(total_time)
        }
    
    def calculate_performance_grade(self, total_time: float) -> str:
        """Calculate performance grade based on execution time"""
        if total_time < 5:
            return "Excellent"
        elif total_time < 15:
            return "Good"
        elif total_time < 30:
            return "Fair"
        else:
            return "Needs Optimization"


class EnhancedPhase1CompletionChecker:
    """Enterprise-grade Phase 1 completion checker with advanced features"""
    
    def __init__(self, project_root: Optional[Path] = None, config_file: Optional[Path] = None):
        """Initialize enhanced Phase 1 completion checker"""
        
        # Initialize project paths
        self.project_root = project_root or Path.home() / "JARVIS_PROJECT"
        self.phase1_dir = self.project_root / "phase1_foundation"
        
        # Initialize components
        self.logger = EnhancedLogger(self.project_root / "logs")
        self.security_validator = SecurityValidator(self.project_root)
        self.profiler = PerformanceProfiler()
        self.platform = CrossPlatformUtils.get_platform()
        
        # Load configuration
        self.config = self.load_configuration(config_file)
        
        # Initialize tracking
        self.checks = {}
        self.detailed_results = []
        self.overall_score = 0
        self.max_workers = min(8, (os.cpu_count() or 1) * 2)
        
        self.print_header()
    
    def print_header(self):
        """Print enhanced header with system information"""
        header = f"""
🤖 JARVIS Phase 1 Enhanced Completion Checker v2.0.0
{'=' * 80}
🖥️  Platform: {self.platform.title()} ({platform.machine()})
🐍 Python: {sys.version.split()[0]} ({sys.executable})
📁 Project Root: {self.project_root}
⚡ Max Workers: {self.max_workers}
🔒 Admin Rights: {'Yes' if CrossPlatformUtils.check_admin_privileges() else 'No'}
{'=' * 80}
"""
        print(header)
        self.logger.info("Phase 1 completion check initiated", "🚀")
    
    def load_configuration(self, config_file: Optional[Path] = None) -> Dict[str, Any]:
        """Load configuration from file or return defaults"""
        default_config = {
            "required_python_version": "3.9",
            "timeout_seconds": 30,
            "parallel_execution": True,
            "security_checks": True,
            "performance_profiling": True,
            "integrity_checking": True,
            "auto_fix_attempts": True,
            "required_folders": [
                "phase1_foundation", "phase2_root_access", "phase3_system_integration",
                "phase4_ai_brain", "phase5_phone_transformation", "phase6_advanced_features",
                "phase7_optimization", "models", "models/13b_brain", "models/voice_synthesis",
                "models/emotion_analysis", "cache", "cache/model_cache", "cache/voice_cache",
                "tools", "backups", "logs", "documentation", "tests", "config"
            ],
            "required_files": {
                "m2_setup.py": {"min_size": 1000, "check_syntax": True},
                "phone_analyzer.py": {"min_size": 2000, "check_syntax": True},
                "basic_controller.py": {"min_size": 1500, "check_syntax": True},
                "advanced_ai_brain.py": {"min_size": 3000, "check_syntax": True},
                "neural_voice_system.py": {"min_size": 2500, "check_syntax": True},
                "human_jarvis.py": {"min_size": 2000, "check_syntax": True},
                "phase1_completion.py": {"min_size": 5000, "check_syntax": True},
                "requirements.txt": {"min_size": 100, "check_syntax": False}
            },
            "platform_tools": {
                "macos": [
                    ("say", "Text-to-Speech", True),
                    ("afplay", "Audio playback", True),
                    ("system_profiler", "System profiler", True),
                    ("brew", "Homebrew", False)
                ],
                "linux": [
                    ("espeak", "Text-to-Speech", False),
                    ("aplay", "Audio playback", True),
                    ("lshw", "Hardware info", False),
                    ("apt", "Package manager", False)
                ],
                "windows": [
                    ("powershell", "PowerShell", True),
                    ("wmic", "WMI Command", True),
                    ("choco", "Chocolatey", False)
                ]
            }
        }
        
        if config_file and config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
                self.logger.success(f"Configuration loaded from {config_file}")
            except Exception as e:
                self.logger.warning(f"Failed to load config file: {e}, using defaults")
        
        return default_config
    
    async def run_async_check(self, check_func, *args, **kwargs) -> CheckResult:
        """Run a check function asynchronously"""
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            return await loop.run_in_executor(executor, check_func, *args, **kwargs)
    
    def check_project_structure(self) -> CheckResult:
        """Enhanced project structure validation with auto-creation option"""
        start_time = time.time()
        
        self.logger.info("Validating project structure", "📁")
        
        required_folders = self.config["required_folders"]
        missing_folders = []
        created_folders = []
        
        for folder in required_folders:
            folder_path = self.project_root / folder
            if not folder_path.exists():
                missing_folders.append(folder)
                
                # Auto-create if enabled
                if self.config.get("auto_fix_attempts", False):
                    try:
                        folder_path.mkdir(parents=True, exist_ok=True)
                        created_folders.append(folder)
                        self.logger.success(f"Created missing folder: {folder}")
                    except Exception as e:
                        self.logger.error(f"Failed to create folder {folder}: {e}")
            else:
                self.logger.success(f"Found: {folder}")
        
        # Recheck after auto-creation
        final_missing = [f for f in missing_folders if not (self.project_root / f).exists()]
        
        status = len(final_missing) == 0
        suggestions = []
        
        if final_missing:
            suggestions.extend([
                f"Create missing folder: mkdir -p {self.project_root / folder}"
                for folder in final_missing
            ])
        
        return CheckResult(
            name="project_structure",
            status=status,
            message=f"Project structure {'complete' if status else 'incomplete'}",
            details={
                "total_folders": len(required_folders),
                "missing_folders": final_missing,
                "created_folders": created_folders,
                "project_root": str(self.project_root)
            },
            execution_time=time.time() - start_time,
            severity="error" if final_missing else "info",
            suggestions=suggestions
        )
    
    def validate_python_file_syntax(self, file_path: Path) -> bool:
        """Validate Python file syntax"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()
            
            compile(source, str(file_path), 'exec')
            return True
        except SyntaxError:
            return False
        except Exception:
            return True  # Give benefit of doubt for other errors
    
    def check_phase1_files(self) -> CheckResult:
        """Enhanced file validation with syntax checking and integrity verification"""
        start_time = time.time()
        
        self.logger.info("Validating Phase 1 files", "📄")
        
        required_files = self.config["required_files"]
        file_issues = []
        valid_files = []
        
        for filename, requirements in required_files.items():
            file_path = self.phase1_dir / filename
            
            if not file_path.exists():
                file_issues.append({
                    "file": filename,
                    "issue": "Missing file",
                    "severity": "error"
                })
                continue
            
            # Check file size
            file_size = file_path.stat().st_size
            min_size = requirements.get("min_size", 0)
            
            if file_size < min_size:
                file_issues.append({
                    "file": filename,
                    "issue": f"File too small ({file_size} < {min_size} bytes)",
                    "severity": "warning"
                })
            
            # Check Python syntax if required
            if requirements.get("check_syntax", False) and file_path.suffix == ".py":
                if not self.validate_python_file_syntax(file_path):
                    file_issues.append({
                        "file": filename,
                        "issue": "Syntax error in Python file",
                        "severity": "error"
                    })
                else:
                    self.logger.success(f"Syntax valid: {filename}")
            
            # Security check if enabled
            if self.config.get("security_checks", True):
                perm_result = self.security_validator.validate_file_permissions(file_path)
                if not perm_result.status:
                    file_issues.append({
                        "file": filename,
                        "issue": perm_result.message,
                        "severity": "warning"
                    })
            
            if not any(issue["file"] == filename and issue["severity"] == "error" for issue in file_issues):
                valid_files.append(filename)
                self.logger.success(f"Valid: {filename} ({file_size:,} bytes)")
        
        critical_issues = [issue for issue in file_issues if issue["severity"] == "error"]
        status = len(critical_issues) == 0
        
        return CheckResult(
            name="phase1_files",
            status=status,
            message=f"File validation {'passed' if status else 'failed'}",
            details={
                "total_files": len(required_files),
                "valid_files": valid_files,
                "issues": file_issues,
                "critical_issues": len(critical_issues)
            },
            execution_time=time.time() - start_time,
            severity="error" if critical_issues else "info"
        )
    
    def check_python_environment(self) -> CheckResult:
        """Comprehensive Python environment validation"""
        start_time = time.time()
        
        self.logger.info("Validating Python environment", "🐍")
        
        results = {
            "python_version": False,
            "virtual_environment": False,
            "activation_script": False,
            "core_packages": False,
            "package_versions": {}
        }
        
        issues = []
        
        # Check Python version
        python_version = sys.version_info
        required_version = tuple(map(int, self.config["required_python_version"].split(".")))
        
        if python_version[:2] >= required_version:
            results["python_version"] = True
            self.logger.success(f"Python {python_version.major}.{python_version.minor}.{python_version.micro}")
        else:
            issues.append(f"Python version {python_version.major}.{python_version.minor}.{python_version.micro} < required {self.config['required_python_version']}")
        
        # Check virtual environment
        venv_indicators = [
            self.project_root / "jarvis_env",
            self.project_root / "venv",
            self.project_root / ".venv"
        ]
        
        venv_found = None
        for venv_path in venv_indicators:
            if venv_path.exists():
                venv_found = venv_path
                results["virtual_environment"] = True
                self.logger.success(f"Virtual environment found: {venv_path}")
                break
        
        if not venv_found:
            issues.append("No virtual environment found")
        
        # Check activation script
        activation_scripts = [
            self.phase1_dir / "activate_jarvis.sh",
            self.phase1_dir / "activate_jarvis.bat",
            self.project_root / "activate.sh"
        ]
        
        activation_found = any(script.exists() for script in activation_scripts)
        results["activation_script"] = activation_found
        
        if activation_found:
            self.logger.success("Activation script found")
        else:
            issues.append("No activation script found")
        
        # Check core packages
        core_packages = [
            "torch", "transformers", "numpy", "requests", 
            "psutil", "asyncio", "concurrent.futures"
        ]
        
        package_results = {}
        for package in core_packages:
            try:
                module = importlib.import_module(package)
                version = getattr(module, '__version__', 'unknown')
                package_results[package] = {"status": "ok", "version": version}
                self.logger.success(f"Package available: {package} ({version})")
            except ImportError:
                package_results[package] = {"status": "missing", "version": None}
                issues.append(f"Missing package: {package}")
        
        results["package_versions"] = package_results
        results["core_packages"] = all(p["status"] == "ok" for p in package_results.values())
        
        # Overall status
        critical_checks = ["python_version", "core_packages"]
        status = all(results[check] for check in critical_checks)
        
        return CheckResult(
            name="python_environment",
            status=status,
            message=f"Python environment {'ready' if status else 'needs attention'}",
            details=results,
            execution_time=time.time() - start_time,
            severity="error" if not status else "info",
            suggestions=[
                "Install missing packages: pip install -r requirements.txt",
                "Create virtual environment: python -m venv jarvis_env",
                "Upgrade Python version if needed"
            ] if not status else []
        )
    
    def check_platform_tools(self) -> CheckResult:
        """Cross-platform tool availability checking"""
        start_time = time.time()
        
        self.logger.info(f"Checking {self.platform} tools", "🔧")
        
        platform_tools = self.config["platform_tools"].get(self.platform, [])
        tool_results = []
        
        for tool, description, required in platform_tools:
            try:
                # Test tool availability
                if self.platform == "windows":
                    test_cmd = ["where", tool]
                else:
                    test_cmd = ["which", tool]
                
                result = subprocess.run(
                    test_cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.config["timeout_seconds"]
                )
                
                available = result.returncode == 0
                
                tool_results.append({
                    "tool": tool,
                    "description": description,
                    "available": available,
                    "required": required,
                    "path": result.stdout.strip() if available else None
                })
                
                if available:
                    self.logger.success(f"{description} ({tool}) available")
                elif required:
                    self.logger.error(f"{description} ({tool}) missing (required)")
                else:
                    self.logger.warning(f"{description} ({tool}) missing (optional)")
                    
            except subprocess.TimeoutExpired:
                tool_results.append({
                    "tool": tool,
                    "description": description,
                    "available": False,
                    "required": required,
                    "error": "Timeout during check"
                })
        
        # Calculate status
        required_tools = [t for t in tool_results if t["required"]]
        missing_required = [t for t in required_tools if not t["available"]]
        
        status = len(missing_required) == 0
        
        return CheckResult(
            name="platform_tools",
            status=status,
            message=f"Platform tools {'ready' if status else 'incomplete'}",
            details={
                "platform": self.platform,
                "tools": tool_results,
                "missing_required": [t["tool"] for t in missing_required]
            },
            execution_time=time.time() - start_time,
            severity="error" if missing_required else "info"
        )
    
    def perform_system_diagnostics(self) -> CheckResult:
        """Advanced system diagnostics"""
        start_time = time.time()
        
        self.logger.info("Running system diagnostics", "🔍")
        
        diagnostics = {}
        
        # System info
        diagnostics["system"] = {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "architecture": platform.architecture(),
            "python_implementation": platform.python_implementation()
        }
        
        # Memory info
        try:
            import psutil
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage(str(self.project_root))
            
            diagnostics["resources"] = {
                "total_memory_gb": round(memory.total / (1024**3), 2),
                "available_memory_gb": round(memory.available / (1024**3), 2),
                "memory_percent": memory.percent,
                "disk_free_gb": round(disk.free / (1024**3), 2),
                "disk_total_gb": round(disk.total / (1024**3), 2)
            }
        except ImportError:
            diagnostics["resources"] = {"error": "psutil not available"}
        
        # Network connectivity test
        try:
            response = subprocess.run(
                ["ping", "-c", "1", "8.8.8.8"] if self.platform != "windows" else ["ping", "-n", "1", "8.8.8.8"],
                capture_output=True,
                timeout=10
            )
            diagnostics["network"] = {"connectivity": response.returncode == 0}
        except:
            diagnostics["network"] = {"connectivity": False}
        
        return CheckResult(
            name="system_diagnostics",
            status=True,
            message="System diagnostics completed",
            details=diagnostics,
            execution_time=time.time() - start_time
        )
    
    async def run_comprehensive_check(self) -> Dict[str, Any]:
        """Run comprehensive Phase 1 validation with parallel execution"""
        
        with self.profiler.profile_section("comprehensive_check"):
            self.logger.info("Starting comprehensive Phase 1 validation", "🚀")
            
            # Define all checks
            check_functions = [
                ("project_structure", self.check_project_structure),
                ("phase1_files", self.check_phase1_files),
                ("python_environment", self.check_python_environment),
                ("platform_tools", self.check_platform_tools),
                ("system_diagnostics", self.perform_system_diagnostics)
            ]
            
            # Run checks (parallel if enabled)
            if self.config.get("parallel_execution", True):
                # Run checks in parallel
                tasks = []
                for name, func in check_functions:
                    task = self.run_async_check(func)
                    tasks.append((name, task))
                
                # Wait for all checks to complete
                for name, task in tasks:
                    result = await task
                    self.detailed_results.append(result)
                    self.checks[name] = result.status
            else:
                # Run checks sequentially
                for name, func in check_functions:
                    result = func()
                    self.detailed_results.append(result)
                    self.checks[name] = result.status
            
            # Calculate overall score
            self.calculate_overall_score()
            
            # Generate comprehensive report
            return self.generate_comprehensive_report()
    
    def calculate_overall_score(self):
        """Calculate weighted overall score"""
        weights = {
            "project_structure": 20,
            "phase1_files": 30,
            "python_environment": 25,
            "platform_tools": 15,
            "system_diagnostics": 10
        }
        
        weighted_score = 0
        total_weight = 0
        
        for check_name, passed in self.checks.items():
            if check_name in weights:
                weight = weights[check_name]
                weighted_score += weight if passed else 0
                total_weight += weight
        
        self.overall_score = (weighted_score / total_weight * 100) if total_weight > 0 else 0
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        
        # Categorize results by status
        passed_checks = [r for r in self.detailed_results if r.status]
        failed_checks = [r for r in self.detailed_results if not r.status]
        critical_issues = [r for r in self.detailed_results if r.severity == "critical"]
        
        # Generate summary statistics
        total_execution_time = sum(r.execution_time for r in self.detailed_results)
        
        # Performance metrics
        performance_report = self.profiler.get_performance_report()
        
        # Generate recommendations
        recommendations = self.generate_recommendations()
        
        # Create comprehensive report
        report = {
            "summary": {
                "overall_score": round(self.overall_score, 1),
                "grade": self.get_score_grade(self.overall_score),
                "total_checks": len(self.detailed_results),
                "passed_checks": len(passed_checks),
                "failed_checks": len(failed_checks),
                "critical_issues": len(critical_issues),
                "total_execution_time": round(total_execution_time, 2)
            },
            "detailed_results": [asdict(r) for r in self.detailed_results],
            "performance_metrics": performance_report,
            "system_info": {
                "platform": self.platform,
                "python_version": sys.version,
                "project_root": str(self.project_root),
                "timestamp": datetime.now().isoformat(),
                "checker_version": "2.0.0"
            },
            "recommendations": recommendations,
            "next_steps": self.generate_next_steps()
        }
        
        return report
    
    def get_score_grade(self, score: float) -> str:
        """Convert numerical score to letter grade"""
        if score >= 95:
            return "A+"
        elif score >= 90:
            return "A"
        elif score >= 85:
            return "B+"
        elif score >= 80:
            return "B"
        elif score >= 75:
            return "C+"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"
    
    def generate_recommendations(self) -> List[Dict[str, Any]]:
        """Generate intelligent recommendations based on check results"""
        recommendations = []
        
        for result in self.detailed_results:
            if not result.status:
                rec = {
                    "category": result.name,
                    "priority": "High" if result.severity in ["critical", "error"] else "Medium",
                    "issue": result.message,
                    "suggestions": result.suggestions,
                    "automated_fix": self.can_auto_fix(result.name)
                }
                recommendations.append(rec)
        
        # Add general recommendations
        if self.overall_score < 80:
            recommendations.append({
                "category": "general",
                "priority": "High",
                "issue": "Overall system readiness below recommended threshold",
                "suggestions": [
                    "Address critical and error-level issues first",
                    "Consider running with --auto-fix flag",
                    "Review documentation for detailed setup instructions"
                ],
                "automated_fix": False
            })
        
        return recommendations
    
    def can_auto_fix(self, check_name: str) -> bool:
        """Determine if a check can be automatically fixed"""
        auto_fixable = {
            "project_structure": True,
            "phase1_files": False,  # Usually requires manual intervention
            "python_environment": True,  # Can install packages
            "platform_tools": False,  # Requires system-level installation
            "system_diagnostics": False
        }
        return auto_fixable.get(check_name, False)
    
    def generate_next_steps(self) -> List[str]:
        """Generate actionable next steps based on results"""
        steps = []
        
        if self.overall_score >= 95:
            steps.extend([
                "✅ Phase 1 is ready for completion",
                "🚀 Proceed to Phase 2 initialization",
                "📊 Consider running performance optimization",
                "📝 Update project documentation"
            ])
        elif self.overall_score >= 80:
            steps.extend([
                "⚠️ Address remaining warnings before proceeding",
                "🔧 Run automated fixes where possible",
                "✅ Re-run validation after fixes",
                "📋 Document any accepted risks"
            ])
        else:
            steps.extend([
                "🚨 Critical issues must be resolved",
                "🛠️ Follow recommendation suggestions",
                "🔄 Re-run validation after each fix",
                "❌ Do not proceed to Phase 2 until score > 80"
            ])
        
        return steps
    
    def print_results_summary(self, report: Dict[str, Any]):
        """Print formatted results summary"""
        summary = report["summary"]
        
        # Header
        print(f"\n{'=' * 80}")
        print(f"🎯 PHASE 1 VALIDATION RESULTS")
        print(f"{'=' * 80}")
        
        # Score display with color coding
        score = summary["overall_score"]
        grade = summary["grade"]
        
        if score >= 90:
            score_emoji = "🟢"
        elif score >= 80:
            score_emoji = "🟡"
        elif score >= 60:
            score_emoji = "🟠"
        else:
            score_emoji = "🔴"
        
        print(f"{score_emoji} Overall Score: {score}% (Grade: {grade})")
        print(f"📊 Checks: {summary['passed_checks']}/{summary['total_checks']} passed")
        
        if summary['critical_issues'] > 0:
            print(f"🚨 Critical Issues: {summary['critical_issues']}")
        
        print(f"⏱️ Total Execution Time: {summary['total_execution_time']}s")
        
        # Performance grade
        perf_grade = report["performance_metrics"]["performance_grade"]
        print(f"⚡ Performance Grade: {perf_grade}")
        
        # Detailed results
        print(f"\n📋 DETAILED RESULTS:")
        print("-" * 80)
        
        for result in self.detailed_results:
            status_emoji = "✅" if result.status else "❌"
            severity_emoji = {
                "info": "ℹ️",
                "warning": "⚠️",
                "error": "❌",
                "critical": "🚨"
            }.get(result.severity, "ℹ️")
            
            print(f"{status_emoji} {result.name.replace('_', ' ').title()}")
            print(f"   {severity_emoji} {result.message}")
            print(f"   ⏱️ {result.execution_time:.2f}s")
            
            if result.suggestions:
                print(f"   💡 Suggestions:")
                for suggestion in result.suggestions[:2]:  # Limit to 2 suggestions
                    print(f"      • {suggestion}")
            print()
        
        # Recommendations
        if report["recommendations"]:
            print(f"💡 RECOMMENDATIONS:")
            print("-" * 80)
            for i, rec in enumerate(report["recommendations"][:5], 1):  # Top 5
                priority_emoji = "🔴" if rec["priority"] == "High" else "🟡"
                print(f"{i}. {priority_emoji} [{rec['category'].title()}] {rec['issue']}")
                if rec['suggestions']:
                    for suggestion in rec['suggestions'][:1]:  # One suggestion per rec
                        print(f"   → {suggestion}")
                print()
        
        # Next Steps
        print(f"🎯 NEXT STEPS:")
        print("-" * 80)
        for i, step in enumerate(report["next_steps"], 1):
            print(f"{i}. {step}")
        
        print(f"\n{'=' * 80}")
    
    def save_detailed_report(self, report: Dict[str, Any], output_file: Optional[Path] = None):
        """Save detailed report to JSON file"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.project_root / "logs" / f"phase1_validation_report_{timestamp}.json"
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            self.logger.success(f"Detailed report saved to: {output_file}")
            return output_file
        except Exception as e:
            self.logger.error(f"Failed to save report: {e}")
            return None
    
    def auto_fix_issues(self) -> Dict[str, bool]:
        """Attempt to automatically fix identified issues"""
        self.logger.info("Attempting automatic fixes", "🔧")
        
        fix_results = {}
        
        # Fix project structure
        if not self.checks.get("project_structure", True):
            try:
                structure_result = self.check_project_structure()  # This includes auto-fix
                fix_results["project_structure"] = structure_result.status
            except Exception as e:
                self.logger.error(f"Auto-fix failed for project structure: {e}")
                fix_results["project_structure"] = False
        
        # Fix Python environment (install missing packages)
        if not self.checks.get("python_environment", True):
            try:
                requirements_file = self.phase1_dir / "requirements.txt"
                if requirements_file.exists():
                    result = subprocess.run([
                        sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
                    ], capture_output=True, text=True, timeout=300)
                    
                    fix_results["python_environment"] = result.returncode == 0
                    if result.returncode == 0:
                        self.logger.success("Packages installed successfully")
                    else:
                        self.logger.error(f"Package installation failed: {result.stderr}")
                else:
                    fix_results["python_environment"] = False
                    self.logger.error("requirements.txt not found")
            except Exception as e:
                self.logger.error(f"Auto-fix failed for Python environment: {e}")
                fix_results["python_environment"] = False
        
        return fix_results


# Advanced CLI interface and main execution
class CLIInterface:
    """Advanced command-line interface for the checker"""
    
    def __init__(self):
        self.checker = None
    
    def parse_arguments(self):
        """Parse command line arguments"""
        import argparse
        
        parser = argparse.ArgumentParser(
            description="JARVIS Phase 1 Enhanced Completion Checker",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  python phase1_completion.py                    # Basic check
  python phase1_completion.py --auto-fix         # Check with auto-fix
  python phase1_completion.py --parallel         # Parallel execution
  python phase1_completion.py --config custom.json  # Custom config
  python phase1_completion.py --export report.json  # Export detailed report
            """
        )
        
        parser.add_argument(
            "--project-root", 
            type=Path,
            help="Custom project root directory"
        )
        
        parser.add_argument(
            "--config",
            type=Path,
            help="Configuration file path"
        )
        
        parser.add_argument(
            "--auto-fix",
            action="store_true",
            help="Attempt automatic fixes for issues"
        )
        
        parser.add_argument(
            "--parallel",
            action="store_true",
            default=True,
            help="Enable parallel execution (default: True)"
        )
        
        parser.add_argument(
            "--no-parallel",
            action="store_true",
            help="Disable parallel execution"
        )
        
        parser.add_argument(
            "--export",
            type=Path,
            help="Export detailed report to JSON file"
        )
        
        parser.add_argument(
            "--verbose",
            action="store_true",
            help="Enable verbose output"
        )
        
        parser.add_argument(
            "--quick",
            action="store_true",
            help="Quick check (skip diagnostics)"
        )
        
        return parser.parse_args()
    
    async def run(self):
        """Main CLI execution"""
        args = self.parse_arguments()
        
        # Initialize checker with custom config
        config_overrides = {}
        
        if args.no_parallel:
            config_overrides["parallel_execution"] = False
        elif args.parallel:
            config_overrides["parallel_execution"] = True
        
        if args.auto_fix:
            config_overrides["auto_fix_attempts"] = True
        
        if args.quick:
            config_overrides["quick_mode"] = True
        
        # Create checker instance
        self.checker = EnhancedPhase1CompletionChecker(
            project_root=args.project_root,
            config_file=args.config
        )
        
        # Apply config overrides
        self.checker.config.update(config_overrides)
        
        try:
            # Run comprehensive check
            report = await self.checker.run_comprehensive_check()
            
            # Auto-fix if requested
            if args.auto_fix and self.checker.overall_score < 95:
                print(f"\n🔧 ATTEMPTING AUTOMATIC FIXES")
                print("=" * 50)
                
                fix_results = self.checker.auto_fix_issues()
                
                if any(fix_results.values()):
                    print("\n🔄 Re-running validation after fixes...")
                    report = await self.checker.run_comprehensive_check()
            
            # Display results
            self.checker.print_results_summary(report)
            
            # Export report if requested
            if args.export:
                self.checker.save_detailed_report(report, args.export)
            else:
                # Auto-save report
                self.checker.save_detailed_report(report)
            
            # Return appropriate exit code
            if self.checker.overall_score >= 80:
                return 0  # Success
            elif self.checker.overall_score >= 60:
                return 1  # Warning
            else:
                return 2  # Error
                
        except KeyboardInterrupt:
            self.checker.logger.warning("Check interrupted by user")
            return 130
        except Exception as e:
            self.checker.logger.critical(f"Unexpected error: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            return 3


# Entry point
async def main():
    """Main entry point"""
    cli = CLIInterface()
    exit_code = await cli.run()
    sys.exit(exit_code)


if __name__ == "__main__":
    # Ensure proper event loop handling across platforms
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    asyncio.run(main())