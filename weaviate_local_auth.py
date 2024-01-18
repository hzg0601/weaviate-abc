import weaviate
import weaviate.classes as wvc
import os
import requests
import json
client = weaviate.connect_to_local(
    host="localhost",
    port=8080,
    grpc_port=50051,
    timeout=(60,60),
    auth_credentials=weaviate.auth.APIKey('pmvdb001')
)
classes_exist = client.collections.list_all()

print(classes_exist)
client.collections.delete_all()
print("！！！！！reconstruction done！！！")

try:
    test_class = client.collections.create(
        name="test_class",
        description="just to test",
        inverted_index_config=wvc.Configure.inverted_index(bm25_b=0.75,bm25_k1=1.2),
        multi_tenancy_config=wvc.Configure.multi_tenancy(enabled=True),
        # replication_config=wvc.Configure.replication(factor=1),
        # 多租户开启时不能shard
        # sharding_config=wvc.Configure.sharding(desired_count=2),
        vector_index_config=wvc.Configure.VectorIndex.hnsw(distance_metric=wvc.VectorDistance.COSINE),
        vectorizer_config=wvc.Configure.Vectorizer.text2vec_transformers(),
    )
except weaviate.exceptions.UnexpectedStatusCodeError as e:
    print(f"WARNING {e}")
    test_class = client.collections.get("test_class")

test_class.tenants.create(tenants=[wvc.Tenant(name='tenant1'),wvc.Tenant(name='tenant2')])

tenant = test_class.with_tenant('tenant1')

req = requests.get("https://raw.githubusercontent.com/weaviate-tutorials/quickstart/main/data/jeopardy_tiny.json")

data = json.loads(req.text)

data_objs = []

for i,d in enumerate(data):
    data_objs.append({
        "answer": d["Answer"],
        "question":d["Question"],
        "category":d["Category"]
    })    

for obj in data_objs:
    tenant.data.insert(obj)

resp = tenant.query.hybrid(query="orange",alpha=0.5,fusion_type=wvc.HybridFusion.RANKED,limit=5,auto_limit=3)
print(resp)

print("-------------------test backup----------------------")
result = client.backup.create(
    backup_id="first-backup-test",
    backend="filesystem",
    include_collections=['Test_class'],
    wait_for_completion=True
)


print(result)

client.close()

