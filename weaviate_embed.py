import os
import weaviate
from weaviate.embedded import EmbeddedOptions
import multiprocessing as mp
from sentence_transformers import SentenceTransformer
def mk_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

persistance_data_path = '/home/pinming/data/weaviate/persistance'
binary_path = '/home/pinming/data/weaviate/binary'
embed_model_path = "/home/pinming/models/bge-large-zh-v1.5"
mk_path(persistance_data_path)
mk_path(binary_path)


def create_instance():
    embeded_options = EmbeddedOptions(
                                    persistence_data_path=persistance_data_path,
                                    binary_path=binary_path,
                                    hostname='127.0.0.1',
                                    port=8003,
                                    grpc_port=50302
                                    )
    client = weaviate.Client(embedded_options=embeded_options)
    return client


# 创建客户端、
def load_data(client):  
    # embeded方式仅支持输入向量，即用户需自己将内容转化为向量。
    # 创建数据集合class
    class_obj = {
                "class":"test_class",
                "vectorizer": "none",#"text2vec-transformers",
                #  "moduleConfig":{
                #      "text2vec-transformers":{
                #          "vectorizeClassName":"false"
                #      }
                #  },
                # KeyError: dataType
                "properties":[
                    {
                        "name":"test class for testing something",
                        "dataType":["text"],
                    }
                ]

                }
    try:
        client.schema.create_class(class_obj)
    except weaviate.exceptions.UnexpectedStatusCodeException as e:
        print("WARNING!!! class_name already exists!")
        print(e)

    embed_model = SentenceTransformer(model_name_or_path=embed_model_path,device="cuda")

    test_texts = ["test","你好"]*100
    pool = embed_model.start_multi_process_pool()
    embeddings = embed_model.encode_multi_process(test_texts,pool=pool,batch_size=100)

    data_objs = [{"corpus": i }for i in test_texts]
    client.batch.configure(batch_size=1000)
    with client.batch as batch:
        for i, data_obj in enumerate(data_objs):
            batch.add_data_object(
                data_obj,
                class_name="test_class",
                vector= embeddings[i]
            )
def do_query(client,query="你是谁"):
    semantic = (
        client.query.get(class_name="test_class",properties=["name"])
        .with_near_test({
            "concepts": ["你好吗"],
            "distance": 0.1,
        #             >>> content = {
        # ...     'concepts': <list of str or str>,
        # ...     # certainty ONLY with `cosine` distance specified in the schema
        # ...     'certainty': <float>, # Optional, either 'certainty' OR 'distance'
        # ...     'distance': <float>, # Optional, either 'certainty' OR 'distance'
        # ...     'moveAwayFrom': { # Optional
        # ...         'concepts': <list of str or str>,
        # ...         'force': <float>
        # ...     },
        # ...     'moveTo': { # Optional
        # ...         'concepts': <list of str or str>,
        # ...         'force': <float>
        # ...     },
        # ...     'autocorrect': <bool>, # Optional
        })
        .with_limit(5)
        .with_offset(1) # paginage分页
        .with_autocut(1) #要将结果限制为与查询距离相似的组，请使用autocut过滤器设置要返回的组数
        .with_additional("distance")
        .with_where({
            "path" : ["id"], #指定过滤的属性
            "operator": "NotEqual", #指定过滤的算子,Equal, And, GreaterThan, LessThan
            "valueString": "e5dc4a4c-ef0f-3aed-89a3-a73435c6bbcf" #valueString,valueInt,value
        })
        .do()
    )
    #!!! BM25 查询字符串在用于使用倒排索引搜索对象之前会被tokenize。
    #!!! 必须在集合定义中为每个属性指定tokenizer,且与vectorizer的tokenizer一致
    #!!! 所以最好还是使用分离的库，额外建一个ES库
    bm25 = (
        client.query.get(class_name="test_class",properties=["name"])
        .with_bm25(query="你好",properties=["name^2"]) #^2表示增加权重
        .with_additional("score")
        .with_limit(3)
        .with_offset(1)
        .with_where({
            "path" : ["id"], #指定过滤的属性
            "operator": "NotEqual", #指定过滤的算子,Equal, And, GreaterThan, LessThan
            "valueString": "e5dc4a4c-ef0f-3aed-89a3-a73435c6bbcf" #valueString,valueInt,value
        })
        .do()

    )

    #
    from weaviate.gql.get import HybridFusion
    hybrid = (
        client.query.get(class_name="test_class",properties=["name"])
        .with_hybrid(
                    query="你好吗",
                     properties=["name^2"],
                     alpha=0.3, # 语义的权重
                     fusion_type=HybridFusion.RELATIVE_SCORE, #融合类型
                    # vector=[1,2,334]

                     )
        .with_additional(["score","explainScore"])
        .with_where({
            "path" : ["id"], #指定过滤的属性
            "operator": "NotEqual", #指定过滤的算子,Equal, And, GreaterThan, LessThan
            "valueString": "e5dc4a4c-ef0f-3aed-89a3-a73435c6bbcf" #valueString,valueInt,value
        })
        .with_limit(3)
        .with_offset(1)
        .with_autocut(1)
        .do()
    )

    rerank_semantic = (
        client.query.get(class_name="test_class",properties=["name"])
        .with_near_text(
            {"concepts": ["你好吗"],
             "distance":0.8}
        )
        .with_additional("rerank(property:'name' query:'好个der'){ score}")
        .with_limit(10)
    )
    rerank_bm25 = (
        client.query.get(class_name="test_class",properties=["name"])
        .with_bm25(query="你好吗",properties=["name"])
        .with_additional("rerank(property: 'name' query: '好个der') { score}")
        .with_limit(10)
        .do()
    )


if __name__ == "__main__":
    mp.freeze_support()
    client = create_instance()
    do_query(client)