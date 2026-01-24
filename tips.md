pip uninstall ipykernel UTF-8 编码：

```python
chcp 65001
```

Create new env：

```python
conda create --name <name> python=3.8
```

Create new env in specified folder：

```python
conda create --prefix C:\Users\25218\Projects\myenv 
```

Activate env and install ipykernel：

```python
pip install ipykernel
conda install -n base nb_conda_kernels # base安装该插件后则不需要执行👇
python -m ipykernel install --user --name howard3.10 --display-name "howard3.10"
```

Upgrade packages：

```python
pip install --upgrade pandas
```

Common requirements

```python
pip install -r requirements.txt
```

Other mirrors 

```python
pip install -r C:\Users\25218\houseprices.txt -i https://mirrors.aliyun.com/pypi/simple/
```

```python
https://mirrors.aliyun.com/pypi/simple/
```

```python
https://pypi.tuna.tsinghua.edu.cn/simple/
```

Check all ipykernal：

```python
jupyter kernelspec list
```

Remove env：

```python
conda env remove --n <A>
```

Remove ipykernal：

```python
jupyter kernelspec remove <kernel_name>
```

Install packages from the official PyPI source：

```python
pip install package -i https://pypi.org/simple
```

Output HTML需要激活所在环境之后：

```python
pip install notebook
```

Switch different disks：

```commonlisp
d：
```

清理

```pyhton
conda clean --all
```

Tensorboard:

```python
tensorboard --logdir=logs路径  #logs是上面指定在writer = SummaryWriter("logs")中指定的文件夹名，日志文件存储在此文件中
tensorboard --logdir==logs --port=XXXX # 可指定端口号
```



```python
init_opts=opts.InitOpts(width="1000px", height="600px")
```



```python
min_ = 40,max_ = 180,dimension=1,
```

Venv

```python
python -m venv myenv

myenv\Scripts\activate

pip install -i https://mirrors.aliyun.com/pypi/simple/ -r requirements.txt
```

特定版本Shap包会和numpy冲突



update packages

```python
pip install --upgrade matplotlib==3.7
```

