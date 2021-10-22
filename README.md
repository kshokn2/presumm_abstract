# presumm_abstract


nvidia-docker run --cpu-shares=2048 --shm-size 50G -it -v /my_local_workspace:/workspace --name abs_pytorch -e LC_ALL=C.UTF-8 --entrypoint=/bin/bash pytorch/pytorch:1.9.0-cuda10.2-cudnn7-devel


git clone https://github.com/SOMJANG/Mecab-ko-for-Google-Colab.git
cd Mecab-ko-for-Google-Colab
ls
bash ./install_mecab-ko_on_colab190912.sh
python -c "from konlpy.tag import Mecab; mecab = Mecab()"
pip install transformers


학습
python main.py -save_dir checkpoint

테스트
python test.py -checkpoint_path checkpoint/model_step_000.pt -submit False
