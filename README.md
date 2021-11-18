# presumm_abstract


nvidia-docker run --cpu-shares=2048 --shm-size 50G -it -v /my_local_workspace:/workspace --name abs_pytorch -e LC_ALL=C.UTF-8 --entrypoint=/bin/bash pytorch/pytorch:1.9.0-cuda10.2-cudnn7-devel
</br></br>

git clone https://github.com/SOMJANG/Mecab-ko-for-Google-Colab.git
</br>
cd Mecab-ko-for-Google-Colab
</br>
ls
</br>
bash ./install_mecab-ko_on_colab190912.sh
</br>
python -c "from konlpy.tag import Mecab; mecab = Mecab()"
</br>
pip install transformers
</br></br>


학습
</br>
python main.py -save_dir checkpoint
</br></br>

테스트
</br>
python test.py -checkpoint_path checkpoint/model_step_000.pt -submit False
