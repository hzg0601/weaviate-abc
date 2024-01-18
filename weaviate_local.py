import weaviate
import weaviate.classes as wvc
import os
import requests
import json


client = weaviate.connect_to_local(
    host="localhost",
    port=8080,
    grpc_port=50051,
    timeout=(60,60)
)
#------------------no multi-tenants-------------------------------------
print("start to test no multi tenants....")
print("*"*100)
try:
    test_class = client.collections.create(
        name="test_class",
        description="a test class",
        vectorizer_config=wvc.Configure.Vectorizer.text2vec_transformers(
            pooling_strategy="masked_mean",vectorize_collection_name=True
            ),
        # bm25_b \in (0,1)越大，文档长度对相关性影响越大，反之越小；
        # bm25_k 制着词频结果在词频饱和度中的上升速度,值越小饱和度变化越快，值越大饱和度变化越慢。
        inverted_index_config=wvc.Configure.inverted_index(bm25_b=0.75,bm25_k1=1.2),
        vector_index_config=wvc.Configure.VectorIndex.hnsw(distance_metric=wvc.VectorDistance.COSINE),
        multi_tenancy_config=wvc.Configure.multi_tenancy(enabled=False),
        sharding_config=wvc.Configure.sharding(virtual_per_physical=1,desired_count=2),
        # init sharding state: not enough replicas: found 1 want 2
        replication_config=wvc.Configure.replication(factor=1),
        # virtual_per_physical在reshard的时候可以减少数据转移时间,但重分片不建议使用，因为会拖慢import速度
        # desired_count即该集合的分片数，默认为节点数
        # actual_count默认等于desired_count，除非在初始化时出问题了
        #! cannot have both shardingConfig and multiTenancyConfig
        
    )
except weaviate.exceptions.UnexpectedStatusCodeError as e:
    print(f"WARNING! {e}")
    test_class = client.collections.get("test_class")

resp = requests.get('https://raw.githubusercontent.com/weaviate-tutorials/quickstart/main/data/jeopardy_tiny.json')
data = json.loads(resp.text)  # Load data

data_objs = []
for i, d in enumerate(data):
    data_objs.append({
        "answer": d["Answer"],
        "question":d["Question"],
        "category":d["Category"]
    })

test_class.data.insert_many(data_objs)
# alpha控制bm25的权重, fusion_type为Relative_Score时，需指定auto_limit,将结果限制为与查询距离相似的组
response = test_class.query.hybrid(query="biology",
                                   alpha=0.75,
                                #    filters=wvc.Filter("answer").is_none(),
                                   limit=5,
                                   fusion_type=wvc.HybridFusion.RANKED,
                                   )
print(response)
client.collections.delete_all()
print("no multi tenants testing done.")
print("*"*100)
#---   --------------------------多租户----------------------------------------------------------------------------
# 多租户是对同一个collection进行隔离，与多账户的概念不同，多租户允许对同一个collection做不同的增删改而互不影响
print("start to test multi tenants....")
print("*"*100)
try:
    multi_class = client.collections.create(
        name="multi_class",
        description="a multi test class",
        vectorizer_config=wvc.Configure.Vectorizer.text2vec_transformers(
            pooling_strategy="masked_mean",vectorize_collection_name=True
            ), # text2vec_transformers的参数并不可选，只能用默认的
        # bm25_b \in (0,1)越大，文档长度对相关性影响越大，反之越小；
        # bm25_k 制着词频结果在词频饱和度中的上升速度,值越小饱和度变化越快，值越大饱和度变化越慢。
        inverted_index_config=wvc.Configure.inverted_index(bm25_b=0.75,bm25_k1=1.2),
        vector_index_config=wvc.Configure.VectorIndex.hnsw(distance_metric=wvc.VectorDistance.COSINE),
        multi_tenancy_config=wvc.Configure.multi_tenancy(enabled=True),
        replication_config=wvc.Configure.replication(factor=1),
        # virtual_per_physical在reshard的时候可以减少数据转移时间；desired_count即该集合的分片数，默认为节点数
        # actual_count默认等于desired_count，除非在初始化时出问题了
        #! cannot have both shardingConfig and multiTenancyConfig
        # sharding_config=wvc.Configure.sharding(virtual_per_physical=2,desired_count=1),
    )
    multi_class.tenants.create(tenants=[wvc.Tenant(name='user1'),wvc.Tenant(name='user2')])
except Exception as e:
    print(e)
    multi_class = client.collections.get("multi_class")

# tenants = multi_class.tenants.get()
# print(tenants)

usr1 = multi_class.with_tenant("user1")

data_id = usr1.data.insert(data_objs[0])
#! multi_tenancy模式不支持insert_many
# data_ids = usr1.data.insert_many(data_objs)

resp = usr1.query.hybrid(
    query="orange",
    alpha=0.5,
    fusion_type=wvc.HybridFusion.RELATIVE_SCORE,
    include_vector=False,
    limit=5,
    auto_limit=3
)
print(resp)
client.collections.delete_all()
print("multi tenants testing done.")
print("*"*100)

print("-------------------test backup----------------------")
result = client.backup.create(
    backup_id="first-backup",
    backend="filesystem",
    include_collections=['test_class',"multi_class"],
    wait_for_completion=True
)

print(result)

client.close()