# neuropixel-pipeline
Schemata and libraries for ingesting and clustering Neuropixel data 

## Using dev-container
For the dev-container you have to be using either VSCode's Dev Container [extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) or its devcontainer-cli (which you can install using the dev container extension).

If you want to play with the schemas in a notebook, it's ideal to set your configuration using an env file (or you can set it at runtime with dj.config). To use the env file put a .env file in the .devcontainer/ folder with the contents of DJ_HOST, DJ_USER, and DJ_PASS env variables and the corresponding values. This file will be ignored by git.

Then just use the Dev Container command "Dev Containers: Reopen in Container".
![Alt text](https://code.visualstudio.com/assets/docs/devcontainers/create-dev-container/dev-containers-reopen.png)
