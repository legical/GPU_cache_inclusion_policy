# 使用（root）
## 编译
```
nvcc -arch sm_86 -lcuda -o testmc multi_cache.cu util.cu
```

## 开启MPS 
```
chmod +x choose_as_root.bash
./choose_as_root.bash
```
输入y

## 运行
```
./testmc
```