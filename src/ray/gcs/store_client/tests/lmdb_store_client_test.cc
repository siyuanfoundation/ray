// Copyright 2025 The Ray Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "ray/gcs/store_client/lmdb_store_client.h"

#include <filesystem>
#include <memory>

#include "ray/gcs/store_client/tests/store_client_test_base.h"

namespace ray {

namespace gcs {

class LMDBStoreClientTest : public StoreClientTestBase {
 public:
  void InitStoreClient() override {
    lmdb_path_ = "lmdb_test_dir";
    std::filesystem::remove_all(lmdb_path_);
    store_client_ =
        std::make_shared<LMDBStoreClient>(*io_service_pool_->Get(), lmdb_path_);
  }

  void TearDown() override {
    StoreClientTestBase::TearDown();
    std::filesystem::remove_all(lmdb_path_);
  }

 protected:
  std::string lmdb_path_;
};

TEST_F(LMDBStoreClientTest, AsyncPutAndAsyncGetTest) { TestAsyncPutAndAsyncGet(); }

TEST_F(LMDBStoreClientTest, AsyncGetAllAndBatchDeleteTest) {
  TestAsyncGetAllAndBatchDelete();
}

TEST_F(LMDBStoreClientTest, TestPersistence) {
  std::string path = "lmdb_persistence_test";
  std::filesystem::remove_all(path);
  {
    auto client = std::make_shared<LMDBStoreClient>(*io_service_pool_->Get(), path);
    std::promise<bool> promise;
    client->AsyncPut("table",
                     "key",
                     "value",
                     true,
                     {[&promise](bool success) { promise.set_value(success); },
                      *io_service_pool_->Get()});
    ASSERT_TRUE(promise.get_future().get());
  }

  // New client with same path should see data
  {
    auto client = std::make_shared<LMDBStoreClient>(*io_service_pool_->Get(), path);
    std::promise<std::optional<std::string>> promise;
    client->AsyncGet(
        "table",
        "key",
        {[&promise](const Status &status, const std::optional<std::string> &data) {
           promise.set_value(data);
         },
         *io_service_pool_->Get()});
    auto result = promise.get_future().get();
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(*result, "value");
  }
  std::filesystem::remove_all(path);
}

}  // namespace gcs

}  // namespace ray

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
