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

#include <sys/stat.h>

#include <filesystem>
#include <future>

#include "ray/util/logging.h"
#include "src/ray/protobuf/gcs.pb.h"

namespace ray::gcs {

namespace {
const std::string kJobCounterTableName = "JobCounter";
const std::string kJobCounterKey = "JobCounterKey";
}  // namespace

LMDBStoreClient::LMDBStoreClient(instrumented_io_context &io_context,
                                 std::string lmdb_path)
    : io_context_(io_context), lmdb_path_(std::move(lmdb_path)) {
  RAY_LOG(INFO) << "Initializing LMDBStoreClient at " << lmdb_path_;

  std::filesystem::create_directories(lmdb_path_);

  RAY_CHECK_EQ(mdb_env_create(&env_), 0);
  // Set a reasonable map size (10GB for production, can be increased).
  RAY_CHECK_EQ(mdb_env_set_mapsize(env_, 10ULL * 1024 * 1024 * 1024), 0);
  // Set max dbs (tables). GCS has around 10-20 tables.
  RAY_CHECK_EQ(mdb_env_set_maxdbs(env_, 128), 0);
  // Set max readers.
  RAY_CHECK_EQ(mdb_env_set_maxreaders(env_, 1024), 0);

  int flags = 0;
  int rc = mdb_env_open(env_, lmdb_path_.c_str(), flags, 0664);
  if (rc != 0) {
    RAY_LOG(FATAL) << "Failed to open LMDB environment at " << lmdb_path_ << ": "
                   << mdb_strerror(rc);
  }

  // Pre-open all known tables in a write transaction.
  MDB_txn *txn;
  RAY_CHECK_EQ(mdb_txn_begin(env_, NULL, 0, &txn), 0);
  for (int i = rpc::TablePrefix_MIN; i <= rpc::TablePrefix_MAX; ++i) {
    if (rpc::TablePrefix_IsValid(i)) {
      std::string name = rpc::TablePrefix_Name(static_cast<rpc::TablePrefix>(i));
      MDB_dbi dbi;
      rc = mdb_dbi_open(txn, name.c_str(), MDB_CREATE, &dbi);
      if (rc == 0) {
        dbis_[name] = dbi;
      }
    }
  }
  std::vector<std::string> extra_tables = {kJobCounterTableName, "test_table"};
  for (const auto &name : extra_tables) {
    MDB_dbi dbi;
    rc = mdb_dbi_open(txn, name.c_str(), MDB_CREATE, &dbi);
    if (rc == 0) {
      dbis_[name] = dbi;
    }
  }
  RAY_CHECK_EQ(mdb_txn_commit(txn), 0);

  // Start worker thread AFTER env is opened.
  worker_thread_ = std::make_unique<std::thread>([this]() {
    auto work_guard = boost::asio::make_work_guard(worker_io_context_);
    worker_io_context_.run();
  });
}

LMDBStoreClient::~LMDBStoreClient() {
  worker_io_context_.stop();
  if (worker_thread_ && worker_thread_->joinable()) {
    worker_thread_->join();
  }
  if (env_) {
    mdb_env_close(env_);
  }
}

MDB_dbi LMDBStoreClient::GetOrCreateTable(MDB_txn *txn, const std::string &table_name) {
  {
    absl::MutexLock lock(&mutex_);
    auto it = dbis_.find(table_name);
    if (it != dbis_.end()) {
      return it->second;
    }
  }

  MDB_dbi dbi;
  int rc = mdb_dbi_open(txn, table_name.c_str(), MDB_CREATE, &dbi);
  if (rc == 0) {
    absl::MutexLock lock(&mutex_);
    dbis_[table_name] = dbi;
    return dbi;
  }

  if (rc == MDB_NOTFOUND || rc == EACCES) {
    return 0;
  }

  RAY_LOG(FATAL) << "Failed to open LMDB database " << table_name << ": "
                 << mdb_strerror(rc) << " (rc=" << rc << ")";
  return 0;
}

void LMDBStoreClient::AsyncPut(const std::string &table_name,
                               const std::string &key,
                               std::string data,
                               bool overwrite,
                               Postable<void(bool)> callback) {
  worker_io_context_.post(
      [this,
       table_name,
       key,
       data = std::move(data),
       overwrite,
       callback = std::move(callback)]() mutable {
        MDB_txn *txn;
        int rc = mdb_txn_begin(env_, NULL, 0, &txn);
        if (rc != 0) {
          RAY_LOG(ERROR) << "Failed to begin LMDB txn: " << mdb_strerror(rc);
          std::move(callback).Post("LMDB.Put", false);
          return;
        }

        MDB_dbi dbi = GetOrCreateTable(txn, table_name);
        MDB_val mdb_key, mdb_data;
        mdb_key.mv_size = key.size();
        mdb_key.mv_data = (void *)key.data();

        bool is_new = true;
        rc = mdb_get(txn, dbi, &mdb_key, &mdb_data);
        if (rc == 0) {
          if (!overwrite) {
            mdb_txn_abort(txn);
            std::move(callback).Post("LMDB.Put", false);
            return;
          }
          is_new = false;
        }

        mdb_data.mv_size = data.size();
        mdb_data.mv_data = (void *)data.data();
        rc = mdb_put(txn, dbi, &mdb_key, &mdb_data, 0);
        if (rc != 0) {
          RAY_LOG(ERROR) << "Failed to put to LMDB: " << mdb_strerror(rc);
          mdb_txn_abort(txn);
          std::move(callback).Post("LMDB.Put", false);
          return;
        }

        rc = mdb_txn_commit(txn);
        if (rc != 0) {
          RAY_LOG(ERROR) << "Failed to commit LMDB txn: " << mdb_strerror(rc);
          std::move(callback).Post("LMDB.Put", false);
          return;
        }

        std::move(callback).Post("LMDB.Put", is_new);
      },
      "LMDB.Put");
}

void LMDBStoreClient::AsyncGet(
    const std::string &table_name,
    const std::string &key,
    ToPostable<rpc::OptionalItemCallback<std::string>> callback) {
  worker_io_context_.post(
      [this, table_name, key, callback = std::move(callback)]() mutable {
        MDB_txn *txn;
        int rc = mdb_txn_begin(env_, NULL, MDB_RDONLY, &txn);
        if (rc != 0) {
          std::move(callback).Post(
              "LMDB.Get", Status::IOError(mdb_strerror(rc)), std::nullopt);
          return;
        }

        MDB_dbi dbi = GetOrCreateTable(txn, table_name);
        if (dbi == 0) {
          mdb_txn_abort(txn);
          std::move(callback).Post("LMDB.Get", Status::OK(), std::nullopt);
          return;
        }

        MDB_val mdb_key, mdb_data;
        mdb_key.mv_size = key.size();
        mdb_key.mv_data = (void *)key.data();

        rc = mdb_get(txn, dbi, &mdb_key, &mdb_data);
        if (rc == MDB_NOTFOUND) {
          mdb_txn_abort(txn);
          std::move(callback).Post("LMDB.Get", Status::OK(), std::nullopt);
          return;
        } else if (rc != 0) {
          mdb_txn_abort(txn);
          std::move(callback).Post(
              "LMDB.Get", Status::IOError(mdb_strerror(rc)), std::nullopt);
          return;
        }

        std::string result((const char *)mdb_data.mv_data, mdb_data.mv_size);
        mdb_txn_abort(txn);
        std::move(callback).Post("LMDB.Get", Status::OK(), std::move(result));
      },
      "LMDB.Get");
}

void LMDBStoreClient::AsyncGetAll(
    const std::string &table_name,
    Postable<void(absl::flat_hash_map<std::string, std::string>)> callback) {
  worker_io_context_.post(
      [this, table_name, callback = std::move(callback)]() mutable {
        absl::flat_hash_map<std::string, std::string> result;
        MDB_txn *txn;
        int rc = mdb_txn_begin(env_, NULL, MDB_RDONLY, &txn);
        if (rc != 0) {
          std::move(callback).Post("LMDB.GetAll", std::move(result));
          return;
        }

        MDB_dbi dbi = GetOrCreateTable(txn, table_name);
        if (dbi == 0) {
          mdb_txn_abort(txn);
          std::move(callback).Post("LMDB.GetAll", std::move(result));
          return;
        }

        MDB_cursor *cursor;
        rc = mdb_cursor_open(txn, dbi, &cursor);
        if (rc != 0) {
          mdb_txn_abort(txn);
          std::move(callback).Post("LMDB.GetAll", std::move(result));
          return;
        }

        MDB_val key, data;
        while ((rc = mdb_cursor_get(cursor, &key, &data, MDB_NEXT)) == 0) {
          result.emplace(std::string((const char *)key.mv_data, key.mv_size),
                         std::string((const char *)data.mv_data, data.mv_size));
        }

        mdb_cursor_close(cursor);
        mdb_txn_abort(txn);
        std::move(callback).Post("LMDB.GetAll", std::move(result));
      },
      "LMDB.GetAll");
}

void LMDBStoreClient::AsyncMultiGet(
    const std::string &table_name,
    const std::vector<std::string> &keys,
    Postable<void(absl::flat_hash_map<std::string, std::string>)> callback) {
  worker_io_context_.post(
      [this, table_name, keys, callback = std::move(callback)]() mutable {
        absl::flat_hash_map<std::string, std::string> result;
        MDB_txn *txn;
        int rc = mdb_txn_begin(env_, NULL, MDB_RDONLY, &txn);
        if (rc != 0) {
          std::move(callback).Post("LMDB.MultiGet", std::move(result));
          return;
        }

        MDB_dbi dbi = GetOrCreateTable(txn, table_name);
        if (dbi != 0) {
          for (const auto &k : keys) {
            MDB_val mdb_key, mdb_data;
            mdb_key.mv_size = k.size();
            mdb_key.mv_data = (void *)k.data();
            if (mdb_get(txn, dbi, &mdb_key, &mdb_data) == 0) {
              result.emplace(
                  k, std::string((const char *)mdb_data.mv_data, mdb_data.mv_size));
            }
          }
        }

        mdb_txn_abort(txn);
        std::move(callback).Post("LMDB.MultiGet", std::move(result));
      },
      "LMDB.MultiGet");
}

void LMDBStoreClient::AsyncDelete(const std::string &table_name,
                                  const std::string &key,
                                  Postable<void(bool)> callback) {
  worker_io_context_.post(
      [this, table_name, key, callback = std::move(callback)]() mutable {
        MDB_txn *txn;
        int rc = mdb_txn_begin(env_, NULL, 0, &txn);
        if (rc != 0) {
          std::move(callback).Post("LMDB.Delete", false);
          return;
        }

        MDB_dbi dbi = GetOrCreateTable(txn, table_name);
        MDB_val mdb_key;
        mdb_key.mv_size = key.size();
        mdb_key.mv_data = (void *)key.data();

        rc = mdb_del(txn, dbi, &mdb_key, NULL);
        if (rc == MDB_NOTFOUND) {
          mdb_txn_abort(txn);
          std::move(callback).Post("LMDB.Delete", false);
          return;
        } else if (rc != 0) {
          mdb_txn_abort(txn);
          std::move(callback).Post("LMDB.Delete", false);
          return;
        }

        mdb_txn_commit(txn);
        std::move(callback).Post("LMDB.Delete", true);
      },
      "LMDB.Delete");
}

void LMDBStoreClient::AsyncBatchDelete(const std::string &table_name,
                                       const std::vector<std::string> &keys,
                                       Postable<void(int64_t)> callback) {
  worker_io_context_.post(
      [this, table_name, keys, callback = std::move(callback)]() mutable {
        MDB_txn *txn;
        int rc = mdb_txn_begin(env_, NULL, 0, &txn);
        if (rc != 0) {
          std::move(callback).Post("LMDB.BatchDelete", 0);
          return;
        }

        MDB_dbi dbi = GetOrCreateTable(txn, table_name);
        int64_t count = 0;
        for (const auto &k : keys) {
          MDB_val mdb_key;
          mdb_key.mv_size = k.size();
          mdb_key.mv_data = (void *)k.data();
          if (mdb_del(txn, dbi, &mdb_key, NULL) == 0) {
            count++;
          }
        }

        mdb_txn_commit(txn);
        std::move(callback).Post("LMDB.BatchDelete", count);
      },
      "LMDB.BatchDelete");
}

void LMDBStoreClient::AsyncGetNextJobID(Postable<void(int)> callback) {
  worker_io_context_.post(
      [this, callback = std::move(callback)]() mutable {
        MDB_txn *txn;
        int rc = mdb_txn_begin(env_, NULL, 0, &txn);
        if (rc != 0) {
          std::move(callback).Post("LMDB.GetNextJobID", -1);
          return;
        }

        MDB_dbi dbi = GetOrCreateTable(txn, kJobCounterTableName);
        MDB_val mdb_key, mdb_data;
        mdb_key.mv_size = kJobCounterKey.size();
        mdb_key.mv_data = (void *)kJobCounterKey.data();

        int job_id = 1;
        rc = mdb_get(txn, dbi, &mdb_key, &mdb_data);
        if (rc == 0) {
          job_id = *(int *)mdb_data.mv_data;
        }

        int next_job_id = job_id + 1;
        mdb_data.mv_size = sizeof(int);
        mdb_data.mv_data = &next_job_id;
        mdb_put(txn, dbi, &mdb_key, &mdb_data, 0);

        mdb_txn_commit(txn);
        std::move(callback).Post("LMDB.GetNextJobID", job_id);
      },
      "LMDB.GetNextJobID");
}

void LMDBStoreClient::AsyncGetKeys(const std::string &table_name,
                                   const std::string &prefix,
                                   Postable<void(std::vector<std::string>)> callback) {
  worker_io_context_.post(
      [this, table_name, prefix, callback = std::move(callback)]() mutable {
        std::vector<std::string> result;
        MDB_txn *txn;
        int rc = mdb_txn_begin(env_, NULL, MDB_RDONLY, &txn);
        if (rc != 0) {
          std::move(callback).Post("LMDB.GetKeys", std::move(result));
          return;
        }

        MDB_dbi dbi = GetOrCreateTable(txn, table_name);
        if (dbi == 0) {
          mdb_txn_abort(txn);
          std::move(callback).Post("LMDB.GetKeys", std::move(result));
          return;
        }

        MDB_cursor *cursor;
        rc = mdb_cursor_open(txn, dbi, &cursor);
        if (rc != 0) {
          mdb_txn_abort(txn);
          std::move(callback).Post("LMDB.GetKeys", std::move(result));
          return;
        }

        MDB_val key, data;
        if (prefix.empty()) {
          rc = mdb_cursor_get(cursor, &key, &data, MDB_FIRST);
        } else {
          key.mv_size = prefix.size();
          key.mv_data = (void *)prefix.data();
          rc = mdb_cursor_get(cursor, &key, &data, MDB_SET_RANGE);
        }

        while (rc == 0) {
          if (!prefix.empty() &&
              (key.mv_size < prefix.size() ||
               memcmp(key.mv_data, prefix.data(), prefix.size()) != 0)) {
            break;
          }
          result.push_back(std::string((const char *)key.mv_data, key.mv_size));
          rc = mdb_cursor_get(cursor, &key, &data, MDB_NEXT);
        }

        mdb_cursor_close(cursor);
        mdb_txn_abort(txn);
        std::move(callback).Post("LMDB.GetKeys", std::move(result));
      },
      "LMDB.GetKeys");
}

void LMDBStoreClient::AsyncExists(const std::string &table_name,
                                  const std::string &key,
                                  Postable<void(bool)> callback) {
  worker_io_context_.post(
      [this, table_name, key, callback = std::move(callback)]() mutable {
        MDB_txn *txn;
        int rc = mdb_txn_begin(env_, NULL, MDB_RDONLY, &txn);
        if (rc != 0) {
          std::move(callback).Post("LMDB.Exists", false);
          return;
        }

        MDB_dbi dbi = GetOrCreateTable(txn, table_name);
        if (dbi == 0) {
          mdb_txn_abort(txn);
          std::move(callback).Post("LMDB.Exists", false);
          return;
        }

        MDB_val mdb_key, mdb_data;
        mdb_key.mv_size = key.size();
        mdb_key.mv_data = (void *)key.data();

        rc = mdb_get(txn, dbi, &mdb_key, &mdb_data);
        mdb_txn_abort(txn);
        std::move(callback).Post("LMDB.Exists", rc == 0);
      },
      "LMDB.Exists");
}

}  // namespace ray::gcs
