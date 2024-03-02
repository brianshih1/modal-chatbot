import modal
import time
import os
import re

stub = modal.Stub("git-repo", image=modal.Image.debian_slim().env({
    "GIT_PYTHON_REFRESH": "quiet"
}).pip_install("GitPython").pip_install("langchain").apt_install("git"))

@stub.function()
def git_clone():
    import git
    git.Repo.clone_from("https://github.com/gitpython-developers/QuickStartTutorialFiles.git", "/code")

def file_filter(path: str):
    # TODO: regex?
    return path.endswith(".md") or path.endswith(".py") or path.endswith(".txt")

def markdown_filter(path: str):
    rule = re.compile(r".+\.md")
    return bool(rule.match(path))

@stub.function()
def load_files(path):
    from langchain_community.document_loaders import GitLoader
    loader = GitLoader(repo_path="./code", clone_url="https://github.com/gitpython-developers/QuickStartTutorialFiles.git", file_filter=file_filter)
    docs = loader.load()
    print(f"Docs: {docs}")
    print("Hello")
    
                    

@stub.local_entrypoint()
def main():
    # git_clone.remote()
    load_files.remote("/code")
    print("foo")
    time.sleep(599)