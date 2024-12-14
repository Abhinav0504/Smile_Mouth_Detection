#ifndef NMS_H
#define NMS_H

#include <vector>
#include <algorithm>
#include "Types.h" // Include for BBox and Detection

// Function declarations
std::vector<Detection> non_maximum_suppression(
        const std::vector<Detection>& detections,
        float min_suppression_threshold,
        float min_score,
        bool weighted = false);

float overlap_similarity(const BBox& box1, const BBox& box2);

std::vector<Detection> non_max_suppression_plain(
        const std::vector<std::pair<int, float>>& indexed_scores,
        const std::vector<Detection>& detections,
        float min_suppression_threshold,
        float min_score);

std::vector<Detection> weighted_non_max_suppression(
        const std::vector<std::pair<int, float>>& indexed_scores,
        const std::vector<Detection>& detections,
        float min_suppression_threshold,
        float min_score);

#endif // NMS_H
