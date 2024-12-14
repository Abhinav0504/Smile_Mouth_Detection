#include "NMS.h"
#include <iostream>
#include <cmath>
#include <limits>

// Main NMS function
std::vector<Detection> non_maximum_suppression(
        const std::vector<Detection>& detections,
        float min_suppression_threshold,
        float min_score,
        bool weighted
) {

    std::cout << std::endl << "\n\tSTEP 7.1: Non-Maximum Suppression main function call" << std::endl;
    std::cout << "\t---------------------------" << std::endl;

    std::cout << "\n\tNMS: Total detections before suppression: " << detections.size() << std::endl;

    // Extract scores and sort detections by score (descending)
    std::vector<std::pair<int, float>> indexed_scores;
    for (size_t i = 0; i < detections.size(); ++i) {
        indexed_scores.emplace_back(i, detections[i].score);
    }
    std::sort(indexed_scores.begin(), indexed_scores.end(),
              [](const std::pair<int, float>& a, const std::pair<int, float>& b) {
                  return a.second > b.second;
              });

    std::cout << "\n\tScores and indices after sorting by score:" << std::endl;
    for (const auto& [index, score] : indexed_scores) {
        std::cout << "\t  Index: " << index << ", Score: " << score << std::endl;
    }
    std::cout << std::endl;




    if (weighted) {
        return weighted_non_max_suppression(indexed_scores, detections, min_suppression_threshold, min_score);
    } else {
        return non_max_suppression_plain(indexed_scores, detections, min_suppression_threshold, min_score);
    }
}

float overlap_similarity(const BBox& box1, const BBox& box2) {
    std::optional<BBox> intersection_opt = box1.intersect(box2);

    float intersect_area = 0.0f;

    // Check if there's an intersection and print the relevant details
    if (intersection_opt.has_value()) {
        BBox intersection = intersection_opt.value();
        intersect_area = intersection.area();
        std::cout << "\n\tOverlap Similarity:" << std::endl;
        std::cout << "\tBox 1: ";
        box1.print(); // Assuming BBox::print exists
        std::cout << "\tBox 2: ";
        box2.print();
        std::cout << "\tIntersection: ";
        intersection.print();
        std::cout << "\tIntersection Area: " << intersect_area << std::endl;
    } else {
        std::cout << "\n\tOverlap Similarity - No intersection between boxes" << std::endl;
        std::cout << "\tBox 1: ";
        box1.print();
        std::cout << "\tBox 2: ";
        box2.print();
    }

    float denominator = box1.area() + box2.area() - intersect_area;
    float iou = (denominator > 0.0f) ? (intersect_area / denominator) : 0.0f;

    // Print final IoU calculation
    std::cout << "\tBox 1 Area: " << box1.area() << std::endl;
    std::cout << "\tBox 2 Area: " << box2.area() << std::endl;
    std::cout << "\tDenominator (Union Area): " << denominator << std::endl;
    std::cout << "\tIoU (Intersection over Union): " << iou << std::endl;

    return iou;
}



// Plain NMS function
std::vector<Detection> non_max_suppression_plain(
        const std::vector<std::pair<int, float>>& indexed_scores,
        const std::vector<Detection>& detections,
        float min_suppression_threshold,
        float min_score
) {
    std::vector<BBox> kept_boxes;
    std::vector<Detection> outputs;

    for (const auto& [index, score] : indexed_scores) {
        // Exit if score is below the threshold
        if (score < min_score) {
            break;
        }

        const Detection& detection = detections[index];
        const BBox& bbox = detection.bbox();

        bool suppressed = false;
        for (const BBox& kept : kept_boxes) {
            if (overlap_similarity(kept, bbox) > min_suppression_threshold) {
                suppressed = true;
                break;
            }
        }

        if (!suppressed) {
            outputs.push_back(detection);
            kept_boxes.push_back(bbox);
        }
    }

    std::cout << "\n\tNMS: Total detections after suppression: " << outputs.size() << std::endl;
    return outputs;
}

// Weighted NMS function
std::vector<Detection> weighted_non_max_suppression(
        const std::vector<std::pair<int, float>>& indexed_scores,
        const std::vector<Detection>& detections,
        float min_suppression_threshold,
        float min_score
) {

    std::cout << std::endl << "\n\tSTEP 7.2: Weighted Non-Maximum Suppression" << std::endl;
    std::cout << "\t---------------------------" << std::endl;

    std::vector<std::pair<int, float>> remaining_indexed_scores = indexed_scores;
    std::vector<Detection> outputs;



    while (!remaining_indexed_scores.empty()) {
        size_t num_prev_indexed_scores = remaining_indexed_scores.size();

        const Detection& detection = detections[remaining_indexed_scores[0].first];
        std::cout << "\n\tProcessing top detection from remaining_indexed_scores:" << std::endl;
        std::cout << "\t\tTop entry in remaining_indexed_scores: ("
                  << remaining_indexed_scores[0].first << ", " << remaining_indexed_scores[0].second << ")" << std::endl;
        std::cout << "\t\tTop detection index: " << remaining_indexed_scores[0].first << std::endl;
        std::cout << "\t\tTop detection score: " << remaining_indexed_scores[0].second << std::endl;

        if (detection.score < min_score) {
            break;
        }

        const BBox& detection_bbox = detection.bbox();
        std::vector<std::pair<int, float>> remaining;
        std::vector<std::pair<int, float>> candidates;
        Detection weighted_detection = detection;




        for (const auto& [index, score] : remaining_indexed_scores) {
            const BBox& remaining_bbox = detections[index].bbox();
            std::cout << "\n\t\tProcessing remaining bbox:" << std::endl;
            std::cout << "\t\t\tIndex: " << index << std::endl;
            std::cout << "\t\t\tScore: " << score << std::endl;
            std::cout << "\t\t\tRemaining BBox: ";
            remaining_bbox.print();  // Assuming `BBox` has a `print()` method

            float similarity = overlap_similarity(detection_bbox, remaining_bbox);

            if (similarity > min_suppression_threshold) {
                candidates.emplace_back(index, score);
            } else {
                remaining.emplace_back(index, score);
            }
        }

        // Print the contents of candidates and remaining scores
        std::cout << "\n\tCandidates after similarity check:" << std::endl;
        for (const auto& [index, score] : candidates) {
            std::cout << "\t\tIndex: " << index << ", Score: " << score << std::endl;
        }

        std::cout << "\n\tRemaining scores after similarity check:" << std::endl;
        for (const auto& [index, score] : remaining) {
            std::cout << "\t\tIndex: " << index << ", Score: " << score << std::endl;
        }

        std::cout << "\tRemaining scores before processing: [";
        for (const auto& [index, score] : remaining_indexed_scores) {
            std::cout << "(" << index << ", " << score << "), ";
        }
        std::cout << "]" << std::endl;

        std::cout << "\tCandidates after similarity check: [";
        for (const auto& [index, score] : candidates) {
            std::cout << "(" << index << ", " << score << "), ";
        }
        std::cout << "]" << std::endl;




        if (!candidates.empty()) {
            std::cout << "\n\tProcessing weighted candidates:" << std::endl;

            size_t keypoints_size = detection.keypoints.size();
            std::vector<cv::Point2f> weighted(keypoints_size, cv::Point2f(0.0f, 0.0f));
            float total_score = 0.0f;

            for (const auto& [index, score] : candidates) {
                total_score += score;
                const auto& candidate_keypoints = detections[index].keypoints;
                for (size_t i = 0; i < candidate_keypoints.size(); ++i) {
                    weighted[i].x += candidate_keypoints[i].x * score;
                    weighted[i].y += candidate_keypoints[i].y * score;
                }
            }
            for (auto& point : weighted) {
                point.x /= total_score;
                point.y /= total_score;
            }

            std::cout << "\tTotal weighted score: " << total_score << std::endl;
            std::cout << "\tWeighted keypoints:" << std::endl;
            for (size_t i = 0; i < weighted.size(); ++i) {
                std::cout << "\t\tKeypoint " << i << ": ("
                          << weighted[i].x << ", " << weighted[i].y << ")" << std::endl;
            }

            weighted_detection = Detection(weighted, detection.score);


        }

        outputs.push_back(weighted_detection);
        remaining_indexed_scores.swap(remaining);

        if (num_prev_indexed_scores == remaining_indexed_scores.size()) {
            break;
        }
    }

    std::cout << "\n\tNMS: Total detections after weighted suppression: " << outputs.size() << std::endl;
    return outputs;
}


