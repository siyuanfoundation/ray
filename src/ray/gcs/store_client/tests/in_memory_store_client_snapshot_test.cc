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

#include <gtest/gtest.h>

#include <fstream>
#include <future>
#include <thread>

#include "ray/common/asio/instrumented_io_context.h"
#include "ray/gcs/store_client/in_memory_store_client.h"

namespace ray::gcs {

class InMemoryStoreClientSnapshotTest : public ::testing::Test {
 protected:
  void SetUp() override {
    snapshot_path_ = "gcs_snapshot_test.bin";
    std::remove(snapshot_path_.c_str());
  }

  void TearDown() override { std::remove(snapshot_path_.c_str()); }

  std::string snapshot_path_;
};

TEST_F(InMemoryStoreClientSnapshotTest, TestSnapshotAndRecovery) {
  instrumented_io_context io_context;
  auto work_guard = boost::asio::make_work_guard(io_context);
  std::thread io_thread([&io_context]() { io_context.run(); });

  {
    auto store_client =
        std::make_unique<InMemoryStoreClient>(io_context, snapshot_path_, 0);

    // Put some data
    std::promise<bool> promise1;
    store_client->AsyncPut(
        "table1",
        "key1",
        "value1",
        true,
        {[&promise1](bool success) { promise1.set_value(success); }, io_context});
    ASSERT_TRUE(promise1.get_future().get());

    std::promise<bool> promise2;
    store_client->AsyncPut(
        "table2",
        "key2",
        "value2",
        true,
        {[&promise2](bool success) { promise2.set_value(success); }, io_context});
    ASSERT_TRUE(promise2.get_future().get());

    // Trigger snapshot
    store_client->TriggerSnapshot();
  }

  // Create a new store client and load snapshot
  {
    auto store_client =
        std::make_unique<InMemoryStoreClient>(io_context, snapshot_path_, 0);
    ASSERT_TRUE(store_client->LoadSnapshot().ok());

    // Verify data
    std::promise<std::optional<std::string>> promise1;
    store_client->AsyncGet(
        "table1",
        "key1",
        {[&promise1](const Status &status, const std::optional<std::string> &data) {
           promise1.set_value(data);
         },
         io_context});
    auto data1 = promise1.get_future().get();
    ASSERT_TRUE(data1.has_value());
    ASSERT_EQ(*data1, "value1");

    std::promise<std::optional<std::string>> promise2;
    store_client->AsyncGet(
        "table2",
        "key2",
        {[&promise2](const Status &status, const std::optional<std::string> &data) {
           promise2.set_value(data);
         },
         io_context});
    auto data2 = promise2.get_future().get();
    ASSERT_TRUE(data2.has_value());
    ASSERT_EQ(*data2, "value2");
  }

  work_guard.reset();
  io_context.stop();
  if (io_thread.joinable()) {
    io_thread.join();
  }
}

TEST_F(InMemoryStoreClientSnapshotTest, TestPeriodicSnapshot) {
  instrumented_io_context io_context;
  auto work_guard = boost::asio::make_work_guard(io_context);
  std::thread io_thread([&io_context]() { io_context.run(); });

  {
    // Snapshot every 100ms
    auto store_client =
        std::make_unique<InMemoryStoreClient>(io_context, snapshot_path_, 100);

    // Put some data
    std::promise<bool> promise;
    store_client->AsyncPut(
        "table",
        "key",
        "value",
        true,
        {[&promise](bool success) { promise.set_value(success); }, io_context});
    ASSERT_TRUE(promise.get_future().get());

    // Wait for at least one snapshot to happen
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
  }

  // Verify file exists and has data
  std::ifstream is(snapshot_path_, std::ios::binary);
  ASSERT_TRUE(is.is_open());
  is.close();

  // Load and verify
  {
    auto store_client =
        std::make_unique<InMemoryStoreClient>(io_context, snapshot_path_, 0);
    ASSERT_TRUE(store_client->LoadSnapshot().ok());

    std::promise<std::optional<std::string>> promise;
    store_client->AsyncGet(
        "table",
        "key",
        {[&promise](const Status &status, const std::optional<std::string> &data) {
           promise.set_value(data);
         },
         io_context});
    auto data = promise.get_future().get();
    ASSERT_TRUE(data.has_value());
    ASSERT_EQ(*data, "value");
  }

  work_guard.reset();
  io_context.stop();
  if (io_thread.joinable()) {
    io_thread.join();
  }
}

}  // namespace ray::gcs

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
