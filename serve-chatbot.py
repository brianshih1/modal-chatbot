import modal
import time


stub = modal.Stub("git-repo", image=modal.Image.debian_slim().env({
    "GIT_PYTHON_REFRESH": "quiet"
}).pip_install("GitPython").apt_install("git"))

@stub.function()
def git_clone():
    import git
    git.Repo.clone_from("https://github.com/gitpython-developers/QuickStartTutorialFiles.git", "./foo")

@stub.local_entrypoint()
def main():
    git_clone.remote()
    print("foo")
    time.sleep(599)