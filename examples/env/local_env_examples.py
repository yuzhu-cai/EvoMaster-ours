"""Local environment usage examples

Demonstrates how to use LocalSession to execute commands and jobs locally.
"""

from evomaster.agent.session import LocalSession, LocalSessionConfig


def example_basic_session():
    """Example 1: Basic Session usage"""
    print("\n" + "=" * 60)
    print("Example 1: Basic Session Usage")
    print("=" * 60)

    with LocalSession() as session:
        # Execute a simple command
        result = session.exec_bash("python --version")
        print(f"Python version: {result['stdout'].strip()}")

        # Execute a multi-line command
        cmd = """
cat > /tmp/test.txt << 'EOF'
Hello, World!
This is a test file.
EOF
cat /tmp/test.txt
"""
        result = session.exec_bash(cmd)
        print(f"\nFile content:\n{result['stdout']}")

        # Check exit code
        print(f"Exit code: {result['exit_code']}")


def example_job_submission():
    """Example 2: Command execution"""
    print("\n" + "=" * 60)
    print("Example 2: Command Execution")
    print("=" * 60)

    with LocalSession() as session:
        # Execute multiple commands
        print("\nExecuting commands...")

        for i in range(3):
            command = f"echo 'Job {i+1}' && sleep 1 && echo 'Job {i+1} completed'"
            result = session.exec_bash(command)
            print(f"  Command {i+1}:")
            print(f"    Exit code: {result['exit_code']}")
            print(f"    Output: {result['stdout'].strip()[:50]}")


def example_file_operations():
    """Example 3: File upload/download"""
    print("\n" + "=" * 60)
    print("Example 3: File Upload/Download")
    print("=" * 60)

    with LocalSession() as session:
        # Create a local file
        import tempfile
        from pathlib import Path

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            local_file = f.name
            f.write("Hello from local file!\n")
            f.write("This is test content.")

        print(f"\nCreated local file: {local_file}")

        # Upload file
        remote_file = "/tmp/uploaded_test.txt"
        session.upload(local_file, remote_file)
        print(f"Uploaded to: {remote_file}")

        # Read the uploaded file
        content = session.read_file(remote_file)
        print(f"\nContent of uploaded file:\n{content}")

        # Download file
        data = session.download(remote_file)
        print(f"Download successful: {len(data)} bytes")

        # Cleanup
        Path(local_file).unlink()


def example_environment_variables():
    """Example 4: Environment variables and working directory"""
    print("\n" + "=" * 60)
    print("Example 4: Environment Variables and Working Directory")
    print("=" * 60)

    config = LocalSessionConfig(
        timeout=30,
        workspace_path="/tmp",
    )

    with LocalSession(config) as session:
        # Execute commands in the specified working directory
        result = session.exec_bash("pwd && ls -la | head -5")
        print(f"\nWorking directory contents:\n{result['stdout']}")

        # Set environment variables and use them
        cmd = """
export MY_VAR="Hello from Environment"
echo "Variable: $MY_VAR"
"""
        result = session.exec_bash(cmd)
        print(f"\nEnvironment variable test:\n{result['stdout']}")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("Local Session (LocalSession) Usage Examples")
    print("=" * 60)

    example_basic_session()
    example_job_submission()
    example_file_operations()
    example_environment_variables()

    print("\n" + "=" * 60)
    print("All examples completed")
    print("=" * 60)


if __name__ == "__main__":
    main()
