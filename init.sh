#!/bin/bash
/opt/conda/envs/continuum/bin/python -m ipykernel install --user --name cntm --display-name "cntm"

sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended

cp ~/init_confs/zshrc ~/.zshrc
mkdir ~/.ssh
cp ~/init_confs/config ~/.ssh/config

cat << 'EOF' >> ~/.zshrc
export "PATH=$PATH:/opt/conda/bin"
EOF

git clone --depth=1 https://github.com/romkatv/powerlevel10k.git ${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/themes/powerlevel10k

# download syntax highlighting extension
git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting

# install autocompletions extension
git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions

git clone --depth 1 https://github.com/junegunn/fzf.git ~/.fzf
~/.fzf/install

sed -i 's/robbyrussell/powerlevel10k\/powerlevel10k/g' ~/.zshrc
cp ~/private/init_confs/p10k.zsh ~/.p10k.zsh
#uv venv && source .venv/bin/activate && uv pip install -e ".[gpu]"
#/opt/conda/envs/ts/bin/python -m ipykernel install --user --name ts --display-name "ts"
/usr/bin/zsh