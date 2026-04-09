#pragma once

// Disable Eigen debug assertions for maximum performance
#define EIGEN_NO_DEBUG
// Enable OpenMP parallelization within Eigen operations
#define EIGEN_USE_OPENMP

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

/**
 * @brief Represents a single node in the binary split tree.
 */
class Node
{
public:
    uint32_t start;
    uint32_t end;
    uint32_t key;

    int32_t parent;
    int32_t left;
    int32_t right;

    float splitValue;
    uint16_t depth;
    bool is_leaf;

    ~Node() = default;

    uint32_t getSize() const { return end - start; }

    Node(int key, int start, int end, int depth = 0, bool is_leaf = false)
        : start(start), end(end), key(key), parent(-1), left(-1), right(-1), splitValue(0), depth(depth), is_leaf(is_leaf) {}
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
    std::vector<Node> nodePool;

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
        nodePool.emplace_back(key, start, end, depth, is_leaf);
        return static_cast<int>(nodePool.size() - 1);
    }

    void deleteSubtree(int node_idx)
    {
        if (node_idx == -1)
            return;

        Node &node = nodePool[node_idx];
        deleteSubtree(node.left);
        deleteSubtree(node.right);

        if (node.parent != -1)
        {
            Node &parent = nodePool[node.parent];
            if (parent.left == node_idx)
                parent.left = -1;
            else if (parent.right == node_idx)
                parent.right = -1;
        }

        if (node_idx == 0)
            root_idx = -1;

        node.left = -1;
        node.right = -1;
        node.parent = -1;
    }

    virtual void splitNode(int node_idx) = 0;

public:
    BinarySplitTreeBase(const Eigen::Ref<const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> &rndm_ref,
                        int index, int tree_depth, int split_depth, float search_space)
        : rndm_projections(rndm_ref), root_idx(-1), treeIndex(index), treeDepth(tree_depth),
          splitDepth(split_depth), offset(treeIndex * (treeDepth + 1)), keyCounter(0), search_space(search_space)
    {
        int max_nodes = (1 << (treeDepth + 1)) - 1;
        nodePool.reserve(max_nodes);

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

    const std::vector<Node> &getNodePool() const { return nodePool; }

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

    void insertNodeAt(int parent_idx, int node_idx)
    {
        Node &parent = nodePool[parent_idx];
        Node &node = nodePool[node_idx];

        if (parent.left == -1)
            parent.left = node_idx;
        else if (parent.right == -1)
            parent.right = node_idx;
        else
            throw std::runtime_error("Both child positions are already occupied");

        node.parent = parent_idx;
        node.depth = parent.depth + 1;
    }

    void printTree(int node_idx, std::string prefix = "", bool isLast = true)
    {
        if (node_idx == -1)
            return;
        Node &node = nodePool[node_idx];

        if (node_idx == root_idx)
            std::cout << "   " << node.key << '\n';
        else
        {
            std::cout << prefix << "|___" << node.key << '\n';
        }

        std::string newPrefix = prefix + (isLast ? "   " : "|   ");
        if (node.left != -1 || node.right != -1)
        {
            if (node.left != -1)
                printTree(node.left, newPrefix, node.right == -1);
            if (node.right != -1)
                printTree(node.right, newPrefix, true);
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

            const Node &node = nodePool[curr];

            if (node.left == -1 && node.right == -1)
                sizes.push_back(node.getSize());
            else
            {
                if (node.right != -1)
                    stack.push_back(node.right);
                if (node.left != -1)
                    stack.push_back(node.left);
            }
        }
    }

    virtual int splitTree(bool approximate_search_space_size)
    {
        if (approximate_search_space_size)
            splitTreeDepth(root_idx);
        else
            splitTreeBag(root_idx);

        nodePool.shrink_to_fit();
        return true_depth;
    }

    void splitTreeDepth(int root_idx)
    {
        std::vector<int> current_leaves;
        current_leaves.push_back(root_idx);
        float threshold = search_space * 1.47;

        for (int i = 1; i < treeDepth; ++i)
        {
            bool stop = true;
            std::vector<int> next_leaves;

            for (int curr_idx : current_leaves)
            {
                if (nodePool[curr_idx].getSize() >= threshold)
                {
                    splitNode(curr_idx);
                    Node &updatedNode = nodePool[curr_idx];

                    if (updatedNode.left != -1 && updatedNode.right != -1)
                    {
                        stop = false;
                        next_leaves.push_back(updatedNode.left);
                        next_leaves.push_back(updatedNode.right);
                    }
                    else
                    {
                        updatedNode.is_leaf = true;
                        true_depth = std::max(true_depth, (int)updatedNode.depth);
                        max_leaf_size = std::max(max_leaf_size, (int)updatedNode.getSize());
                    }
                }
                else
                {
                    true_depth = std::max(true_depth, (int)nodePool[curr_idx].depth);
                    max_leaf_size = std::max(max_leaf_size, (int)nodePool[curr_idx].getSize());
                }
            }

            current_leaves = next_leaves;
            if (stop || current_leaves.empty())
                break;
        }

        for (int curr_idx : current_leaves)
        {
            Node &node = nodePool[curr_idx];
            true_depth = std::max(true_depth, (int)node.depth);
            max_leaf_size = std::max(max_leaf_size, (int)node.getSize());
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

            if (nodePool[curr_idx].getSize() <= target_leaf_size)
            {
                nodePool[curr_idx].is_leaf = true;
                true_depth = std::max(true_depth, (int)nodePool[curr_idx].depth);
                max_leaf_size = std::max(max_leaf_size, (int)nodePool[curr_idx].getSize());
                continue;
            }

            splitNode(curr_idx);
            Node &node = nodePool[curr_idx];

            if (node.right != -1 && node.left != -1)
            {
                st.push_back(node.right);
                st.push_back(node.left);
            }
            else
            {
                true_depth = std::max(true_depth, (int)node.depth);
                max_leaf_size = std::max(max_leaf_size, (int)node.getSize());
            }
        }
    }

    FORCE_INLINE Node &findLeafForQuery(const float *__restrict q)
    {
        int idx = root_idx;
        Node *__restrict nodes = nodePool.data();
        const int baseOffset = offset;

        for (;;)
        {
            Node &cur = nodes[idx];

            const int left = cur.left;
            const int right = cur.right;
            if (left == -1 && right == -1)
                return cur;

            const int d = cur.depth;
            const float splitValue = cur.splitValue;

            const float v = static_cast<float>(static_cast<Scalar>(q[baseOffset + d]));

            const int preferred_idx = (v < splitValue) ? left : right;
            const int other_idx = (v < splitValue) ? right : left;

            if (preferred_idx != -1)
            {
                idx = preferred_idx;
                continue;
            }

            if (other_idx != -1)
            {
                idx = other_idx;
                continue;
            }

            return cur;
        }
    }

    inline std::span<const unsigned int> getLeafIndices(const float *__restrict projectedQuery) noexcept
    {
        Node &leaf = findLeafForQuery(projectedQuery);
        return {indices.data() + leaf.start, leaf.getSize()};
    }
};