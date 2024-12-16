
# 修改graphrag适配国内大模型

## 方案
### 并行逻辑
- 开发一个并行openai调用的逻辑，-> LLMParamter() 修改成员变量， embedding存储api_key
- 接口适配的修改

### A->B->C, call_llm()
- call_llm： 实际调用大模型之前，我直接截胡
- 初始化，环境变量的读取，llm实例的注册，通过全局变量保存大模型llm实例，避免每次调用都要生成实例
- factories, 实例化不同平台的llm实例

## 适配国内大模型，在哪些环节上修改
- settings.yaml文件的修改, 模型名称配置：哪个平台，哪个模型
- .env
- llm,embed调用期间适配国内大模型的修改
- extra：支持平台的管理，api_key的管理，factories


github: graphrag-more

