#!/usr/bin/env python3
"""
Troubleshoot LangGraph deployment issues
"""

import os
import json
import subprocess
import sys
from pathlib import Path

def check_files():
    """Check if all required files exist"""
    print("📁 Checking required files...")
    
    required_files = [
        "langgraph.json",
        "pyproject.toml", 
        "requirements.txt",
        "src/__init__.py",
        "src/config.py",
        "src/ingest_graph.py",
        "src/query_graph.py",
        "src/main.py"
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
            print(f"❌ Missing: {file}")
        else:
            print(f"✅ Found: {file}")
    
    return len(missing_files) == 0

def check_langgraph_json():
    """Validate langgraph.json structure"""
    print("\n📋 Checking langgraph.json...")
    
    try:
        with open("langgraph.json", "r") as f:
            config = json.load(f)
        
        # Check required fields
        required_fields = ["dependencies", "graphs"]
        for field in required_fields:
            if field not in config:
                print(f"❌ Missing field: {field}")
                return False
            else:
                print(f"✅ Found field: {field}")
        
        # Check graphs
        graphs = config.get("graphs", {})
        if not graphs:
            print("❌ No graphs defined")
            return False
        
        for graph_name, graph_path in graphs.items():
            print(f"📊 Graph: {graph_name} -> {graph_path}")
            
            # Parse module path
            try:
                module, attr = graph_path.split(":")
                module_file = module.replace(".", "/") + ".py"
                if not Path(module_file).exists():
                    print(f"❌ Graph module not found: {module_file}")
                    return False
                print(f"✅ Graph module exists: {module_file}")
            except ValueError:
                print(f"❌ Invalid graph path format: {graph_path}")
                return False
        
        print("✅ langgraph.json structure is valid")
        return True
        
    except FileNotFoundError:
        print("❌ langgraph.json not found")
        return False
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in langgraph.json: {e}")
        return False

def check_python_syntax():
    """Check Python files for syntax errors"""
    print("\n🐍 Checking Python syntax...")
    
    python_files = [
        "src/config.py",
        "src/ingest_graph.py", 
        "src/query_graph.py",
        "src/main.py"
    ]
    
    all_valid = True
    for file in python_files:
        if Path(file).exists():
            try:
                with open(file, 'r') as f:
                    code = f.read()
                compile(code, file, 'exec')
                print(f"✅ Syntax OK: {file}")
            except SyntaxError as e:
                print(f"❌ Syntax Error in {file}: {e}")
                all_valid = False
        else:
            print(f"❌ File not found: {file}")
            all_valid = False
    
    return all_valid

def check_imports():
    """Check if imports work"""
    print("\n📦 Checking imports...")
    
    try:
        sys.path.insert(0, "src")
        
        # Test config
        try:
            import config
            print("✅ config.py imports successfully")
        except Exception as e:
            print(f"❌ config.py import error: {e}")
            return False
        
        # Test graphs
        try:
            from ingest_graph import graph_ingest
            print("✅ ingest_graph imports successfully")
        except Exception as e:
            print(f"❌ ingest_graph import error: {e}")
            return False
        
        try:
            from query_graph import graph_query
            print("✅ query_graph imports successfully")
        except Exception as e:
            print(f"❌ query_graph import error: {e}")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ General import error: {e}")
        return False

def check_environment():
    """Check environment variables"""
    print("\n🔧 Checking environment...")
    
    # Check .env file
    if Path(".env").exists():
        print("✅ .env file found")
        try:
            with open(".env", "r") as f:
                env_content = f.read()
            
            required_vars = ["OPENAI_API_KEY", "PINECONE_API_KEY"]
            for var in required_vars:
                if var in env_content and not env_content.count(f"{var}=your_") > 0:
                    print(f"✅ {var} appears to be set")
                else:
                    print(f"⚠️ {var} may not be properly set")
        except:
            print("❌ Could not read .env file")
    else:
        print("⚠️ No .env file found")

def check_langgraph_cli():
    """Check if langgraph CLI is installed and working"""
    print("\n🛠️ Checking LangGraph CLI...")
    
    try:
        result = subprocess.run(["langgraph", "--version"], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"✅ LangGraph CLI installed: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ LangGraph CLI error: {result.stderr}")
            return False
    except FileNotFoundError:
        print("❌ LangGraph CLI not found. Install with: pip install langgraph-cli")
        return False
    except subprocess.TimeoutExpired:
        print("❌ LangGraph CLI command timed out")
        return False

def run_deployment_test():
    """Test deployment command"""
    print("\n🚀 Testing deployment command...")
    
    try:
        # Try langgraph dev first to see if it works locally
        print("Testing local development server...")
        result = subprocess.run(["langgraph", "dev", "--help"], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("✅ langgraph dev command works")
        else:
            print(f"❌ langgraph dev error: {result.stderr}")
            return False
            
        # Check if we can parse the project
        result = subprocess.run(["langgraph", "dev", "--dry-run"], 
                              capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ Project structure is valid for deployment")
            return True
        else:
            print(f"❌ Project validation failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Deployment test timed out")
        return False
    except Exception as e:
        print(f"❌ Deployment test error: {e}")
        return False

def generate_fixes():
    """Generate common fixes"""
    print("\n🔧 Common Fixes:")
    print("=" * 40)
    
    print("1. Install LangGraph CLI:")
    print("   pip install langgraph-cli")
    
    print("\n2. Login to LangGraph:")
    print("   langgraph login")
    
    print("\n3. Fix langgraph.json if needed:")
    print('   Ensure it has "dependencies" and "graphs" fields')
    
    print("\n4. Set environment variables:")
    print("   Create .env with OPENAI_API_KEY and PINECONE_API_KEY")
    
    print("\n5. Test locally first:")
    print("   langgraph dev")
    
    print("\n6. Deploy:")
    print("   langgraph deploy")

def main():
    print("🔍 LangGraph Deployment Troubleshooter")
    print("=" * 50)
    
    # Run all checks
    checks = [
        ("Files", check_files),
        ("LangGraph JSON", check_langgraph_json), 
        ("Python Syntax", check_python_syntax),
        ("Imports", check_imports),
        ("Environment", check_environment),
        ("LangGraph CLI", check_langgraph_cli),
    ]
    
    results = {}
    for name, check_func in checks:
        results[name] = check_func()
    
    # Summary
    print("\n📊 SUMMARY")
    print("=" * 30)
    
    all_passed = True
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All checks passed! Try deploying:")
        print("   langgraph deploy")
    else:
        print("\n⚠️ Some checks failed. See fixes below:")
        generate_fixes()

if __name__ == "__main__":
    main()