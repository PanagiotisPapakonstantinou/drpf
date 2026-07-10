#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(_MSC_VER)
#define FORCE_INLINE __forceinline
#else
#define FORCE_INLINE inline __attribute__((always_inline))
#endif

#ifndef PREFETCH
#if defined(__GNUC__) || defined(__clang__)
#define PREFETCH(addr) __builtin_prefetch(addr)
#elif defined(_MSC_VER)
#include <xmmintrin.h>
#define PREFETCH(addr) _mm_prefetch(reinterpret_cast<const char *>(addr), _MM_HINT_T0)
#else
#define PREFETCH(addr)
#endif
#endif

namespace drpf
{

    /**
     * @brief 8-byte aligned routing node. Perfectly fits 8 nodes per 64-byte cache line.
     * Contains ONLY the data needed during tree traversal.
     */
    struct RoutingNode
    {
        float splitValue;
        int32_t left;
    };

    /**
     * @brief Cold data used only during construction and at the final leaf.
     */
    struct LeafData
    {
        uint32_t start;
        uint32_t end;
        uint32_t key;
        int32_t parent;
        uint16_t depth;

        uint32_t getSize() const { return end - start; }
    };

    /**
     * @brief Abstract Base Class for a Random Projection Tree.
     * Handles tree construction, indexing, and memory pooling.
     */
    template <typename Scalar = Eigen::half>
    class BinarySplitTreeBase
    {
    protected:
        const Eigen::Ref<const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> rndm_projections;
        std::vector<uint32_t> indices;

        std::vector<RoutingNode> routingPool;
        std::vector<LeafData> leafDataPool;

        int root_idx;
        int treeIndex;
        int treeDepth;
        int splitDepth;
        int true_depth;
        int max_leaf_size = 0;

        int offset;
        int keyCounter;
        float search_space;

        int allocNode(int key, int start, int end, int depth = 0, bool is_leaf = false)
        {
            RoutingNode rNode = {0.0f, -1};
            LeafData lData = {static_cast<uint32_t>(start), static_cast<uint32_t>(end), static_cast<uint32_t>(key), -1, static_cast<uint16_t>(depth)};

            routingPool.push_back(rNode);
            leafDataPool.push_back(lData);

            return static_cast<int>(routingPool.size() - 1);
        }

        void deleteSubtree(int node_idx)
        {
            if (node_idx == -1)
                return;

            RoutingNode &rNode = routingPool[node_idx];
            LeafData &lData = leafDataPool[node_idx];

            if (rNode.left != -1)
            {
                deleteSubtree(rNode.left);
                deleteSubtree(rNode.left + 1);
            }

            if (lData.parent != -1)
            {
                RoutingNode &parentR = routingPool[lData.parent];
                if (parentR.left == node_idx || (parentR.left + 1) == node_idx)
                {
                    parentR.left = -1;
                }
            }

            if (node_idx == root_idx)
                root_idx = -1;

            rNode.left = -1;
            lData.parent = -1;
        }

        virtual void splitNode(int node_idx) = 0;

    public:
        BinarySplitTreeBase(const Eigen::Ref<const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> &rndm_ref,
                            int index, int tree_depth, int split_depth, float search_space)
            : rndm_projections(rndm_ref), root_idx(-1), treeIndex(index), treeDepth(tree_depth),
              splitDepth(split_depth), offset(treeIndex * (treeDepth + 1)), keyCounter(0), search_space(search_space)
        {
            int max_nodes = (1 << (treeDepth + 1)) - 1;
            routingPool.reserve(max_nodes);
            leafDataPool.reserve(max_nodes);

            indices.resize(rndm_projections.rows());
            std::iota(indices.begin(), indices.end(), 0);

            true_depth = 0;
            initialiseRoot(keyCounter, 0, static_cast<int>(indices.size()));
        }

        virtual ~BinarySplitTreeBase()
        {
            deleteSubtree(root_idx);
        }

        void updateOffset(int new_depth)
        {
            const_cast<int &>(this->offset) = this->treeIndex * (new_depth + 1);
            const_cast<int &>(this->treeDepth) = new_depth;
        }

        const std::vector<RoutingNode> &getRoutingPool() const { return routingPool; }
        const std::vector<LeafData> &getLeafDataPool() const { return leafDataPool; }
        const std::vector<uint32_t> &getIndices() const { return indices; }

        int getRootIndex() const { return root_idx; }
        int getOffset() const { return offset; }
        int getMaxLeafSize() const { return max_leaf_size; }

        void initialiseRoot(int key, int start, int end)
        {
            if (root_idx != -1)
                throw std::runtime_error("Root already initialized");
            root_idx = allocNode(key, start, end);
        }

        std::pair<int, int> allocChildren(int parent_idx, int left_key, int left_start, int left_end,
                                          int right_key, int right_start, int right_end)
        {
            RoutingNode &parentR = routingPool[parent_idx];

            int depth = leafDataPool[parent_idx].depth + 1;

            int left_idx = allocNode(left_key, left_start, left_end, depth);
            int right_idx = allocNode(right_key, right_start, right_end, depth);

            parentR.left = left_idx;

            leafDataPool[left_idx].parent = parent_idx;
            leafDataPool[right_idx].parent = parent_idx;

            return {left_idx, right_idx};
        }

        void printTree(int node_idx, std::string prefix = "", bool isLast = true)
        {
            if (node_idx == -1)
                return;

            RoutingNode &rNode = routingPool[node_idx];
            LeafData &lData = leafDataPool[node_idx];

            if (node_idx == root_idx)
                std::cout << "   " << lData.key << '\n';
            else
                std::cout << prefix << "|___" << lData.key << '\n';

            std::string newPrefix = prefix + (isLast ? "   " : "|   ");

            if (rNode.left != -1)
            {
                printTree(rNode.left, newPrefix, false);
                printTree(rNode.left + 1, newPrefix, true);
            }
        }

        void collectLeafSizes(std::vector<int> &sizes) const
        {
            if (root_idx == -1)
                return;

            std::vector<int> stack;
            stack.reserve(64);
            stack.push_back(root_idx);

            while (!stack.empty())
            {
                int curr = stack.back();
                stack.pop_back();

                const RoutingNode &rNode = routingPool[curr];

                if (rNode.left == -1)
                    sizes.push_back(leafDataPool[curr].getSize());
                else
                {
                    stack.push_back(rNode.left + 1);
                    stack.push_back(rNode.left);
                }
            }
        }

        virtual int splitTree(bool approximate_search_space_size)
        {
            if (approximate_search_space_size)
                splitTreeDepth(root_idx);
            else
                splitTreeBag(root_idx);

            routingPool.shrink_to_fit();
            leafDataPool.shrink_to_fit();
            return true_depth;
        }

        void splitTreeDepth(int root_idx)
        {
            std::vector<int> current_leaves;
            current_leaves.push_back(root_idx);
            float threshold = search_space * 1.47f;

            for (int i = 1; i < treeDepth; ++i)
            {
                bool stop = true;
                std::vector<int> next_leaves;

                for (int curr_idx : current_leaves)
                {
                    if (leafDataPool[curr_idx].getSize() >= threshold)
                    {
                        splitNode(curr_idx);
                        RoutingNode &rNode = routingPool[curr_idx];

                        if (rNode.left != -1)
                        {
                            stop = false;
                            next_leaves.push_back(rNode.left);
                            next_leaves.push_back(rNode.left + 1);
                        }
                        else
                        {
                            true_depth = std::max(true_depth, (int)leafDataPool[curr_idx].depth);
                            max_leaf_size = std::max(max_leaf_size, (int)leafDataPool[curr_idx].getSize());
                        }
                    }
                    else
                    {
                        true_depth = std::max(true_depth, (int)leafDataPool[curr_idx].depth);
                        max_leaf_size = std::max(max_leaf_size, (int)leafDataPool[curr_idx].getSize());
                    }
                }

                current_leaves = next_leaves;
                if (stop || current_leaves.empty())
                    break;
            }

            for (int curr_idx : current_leaves)
            {
                true_depth = std::max(true_depth, (int)leafDataPool[curr_idx].depth);
                max_leaf_size = std::max(max_leaf_size, (int)leafDataPool[curr_idx].getSize());
            }
        }

        void splitTreeBag(int root_idx)
        {
            std::vector<int> st;
            st.reserve(treeDepth + 1);
            st.push_back(root_idx);

            int rows = rndm_projections.rows();
            float max_threshold = 1.47f;
            float target_leaf_size = max_threshold * (rows / std::pow(2.0f, splitDepth));

            while (!st.empty())
            {
                int curr_idx = st.back();
                st.pop_back();

                if (leafDataPool[curr_idx].getSize() <= target_leaf_size)
                {
                    true_depth = std::max(true_depth, (int)leafDataPool[curr_idx].depth);
                    max_leaf_size = std::max(max_leaf_size, (int)leafDataPool[curr_idx].getSize());
                    continue;
                }

                splitNode(curr_idx);
                RoutingNode &rNode = routingPool[curr_idx];

                if (rNode.left != -1)
                {
                    st.push_back(rNode.left + 1);
                    st.push_back(rNode.left);
                }
                else
                {
                    true_depth = std::max(true_depth, (int)leafDataPool[curr_idx].depth);
                    max_leaf_size = std::max(max_leaf_size, (int)leafDataPool[curr_idx].getSize());
                }
            }
        }

        FORCE_INLINE int findLeafIdxForQuery(const float *__restrict q) const
        {
            int idx = root_idx;
            const RoutingNode *__restrict nodes = routingPool.data();

            int current_depth = 0;

            while (nodes[idx].left != -1)
            {
                const RoutingNode &cur = nodes[idx];

                PREFETCH(&nodes[cur.left]);

                const float v = q[offset + current_depth];

                idx = cur.left + (v >= cur.splitValue);

                current_depth++;
            }

            return idx;
        }

        inline std::span<const uint32_t> getLeafIndices(const float *__restrict projectedQuery) const noexcept
        {
            int leaf_idx = findLeafIdxForQuery(projectedQuery);
            const LeafData &leaf = leafDataPool[leaf_idx];
            return {indices.data() + leaf.start, leaf.getSize()};
        }
    };

}