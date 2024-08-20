# python import custom modules notes

## Example directory
```
.
├── app
│   ├── __init__.py
│   │ 
│   ├── folder
│   │   └── hi2.py
│   ├── my_models
│   │   └── hi.py
│   └── test.py
└── tree.txt

```
app 底下要有__init__.py
其他folder不用, 除非它有機會被人直接import

## whats does the above means?
app是我的custom module



## how to import?
### e.g.1
在hi2.py import class `Hi` in hi.py

**做法1** (relative import)

```
# hi2.py
from ..my_models.hi import Hi
```

**做法2** (absolute import)
```
# hi2.py
from app.my_models.hi import Hi
```

**how to run hi2.py**
```
# 假設現在在hi2.py位處的資料夾
cd ../../  # cd 至root
python -m app.folder.hi2
```
- `-m` flag means run as a module
- ❗❗注意: 不用.py!!!
- chat gpt said:
  - When using -m, Python treats the specified module as part of a package and sets up the sys.path accordingly

### e.g.2
在test.py import class `Hi` in hi.py

**做法1** (absolute import)
```
# test.py
from app.my_models.hi import Hi
```
- how to run:
  - cd to root (包住app folder 的folder)
  - run `python -m app.test`

**做法2** (直接用absolute import, 不用module)

```
# test.py
from my_models.hi import Hi
```
- how to run:
  - cd  into app, run `python test.py` (run as standalone module)

**做法3** 

treat `my_models` as a module
```
# test.py
from my_models.hi import Hi
```
在my_models 底下加`__init__.py`
- cd to app (包住my_models 的folder) (必須❗)
- run `python -m test`
