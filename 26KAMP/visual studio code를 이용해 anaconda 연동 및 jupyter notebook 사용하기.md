---
layout: single
title: "visual studio code를 이용해 anaconda 연동 및 jupyter notebook 사용하기"
categories:
  - "26KAMP"
author_profile: true
toc: true
toc_sticky: true
permalink: /26KAMP/vs code로 anaconda 연동하기.md
---

# 1. 가상환경 설정
## 1.1. Anaconda prompt 를 관리자 권한으로 실행

<p align="center">
  <a href="#">
    <img src="/images/vscode/figure1_1.png" style="width: min(100%, 860px); height: auto;" />
  </a>
  <br>
</p>


## 1.2. 가상환경 생성
conda create -n {가상환경 이름} python={버전} -y (개인적으로는 3.12.7을 이용) 

<p align="center">
  <a href="#">
    <img src="/images/vscode/figure1_2.png" style="width: min(100%, 860px); height: auto;" />
  </a>
  <br>
</p>


## 1.3. 가상환경 활성화
conda env list 입력해 가상환경이 생성되었는지 확인
conda activate {가상환경이름}

<p align="center">
  <a href="#">
    <img src="/images/vscode/figure1_3.png" style="width: min(100%, 860px); height: auto;" />
  </a>
  <br>
</p>
<p align="center">
  <a href="#">
    <img src="/images/vscode/figure1_4.png" style="width: min(100%, 860px); height: auto;" />
  </a>
  <br>
</p>


## 1.4. ipykernel 설치
pip install ipykernel

<p align="center">
  <a href="#">
    <img src="/images/vscode/figure1_5.png" style="width: min(100%, 860px); height: auto;" />
  </a>
  <br>
</p>


## 1.5. jupyter kernel에 가상환경 마운트
$python -m ipykernel install --user --name {가상환경 이름} --display-name {jupyter에서 보이게 할 이름}

<p align="center">
  <a href="#">
    <img src="/images/vscode/figure1_6.png" style="width: min(100%, 860px); height: auto;" />
  </a>
  <br>
</p>


## 1.6. 가상환경 비활성화
conda deactivate

<p align="center">
  <a href="#">
    <img src="/images/vscode/figure1_7.png" style="width: min(100%, 860px); height: auto;" />
  </a>
  <br>
</p>


# 2. 가상환경 연결
## 2.1. visual studio code extension에서 python 검색 -> python extenstion pack 설치
<p align="center">
  <a href="#">
    <img src="/images/vscode/figure2_1.png" style="width: min(100%, 960px); height: auto;" />
  </a>
  <br>
</p>


## 2.2. anaconda prompt 상에서 jupyter notebook 입력 및 주소 확인
가상환경에 접속된 상태로는 jupyter notebook 실행 불가
반대로 jupyter notebook이 실행된 상태에서는 프롬프트 상에서 가상환경 접속 불가능
<p align="center">
  <a href="#">
    <img src="/images/vscode/figure2_2.png" style="width: min(100%, 860px); height: auto;" />
  </a>
  <br>
</p>


## 2.3. .ipynb 파일 생성
vs code의 아이콘을 눌러 .ipynb 파일 생성

<p align="center">
  <a href="#">
    <img src="/images/vscode/figure2_3.png" style="width: min(100%, 620px); height: auto;" />
  </a>
  <br>
</p>


## 2.4. 1에서 만든 가상환경(커널) 선택
화면 우상단의 커널을 눌러 기존에 만든 가상환경 선택
<p align="center">
  <a href="#">
    <img src="/images/vscode/figure2_4.png" style="width: min(100%, 760px); height: auto;" />
  </a>
  <br>
</p>


## 2.5. 코드 실행
단축키(a 또는 b)나 마우스 우클릭으로 셀을 생성 후 코드 실행
