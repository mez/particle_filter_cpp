/*
 * particle_filter.cpp
 *
 *  Created on: May 29, 2017
 *      Author: Mez Gebre
 */

#include <random>
#include <sstream>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  num_particles = 10;

  default_random_engine gen;
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  const double init_weight = 1.0;

  for (int i = 0; i < num_particles; ++i) {
    particles.push_back(Particle {i, dist_x(gen), dist_y(gen), dist_theta(gen), init_weight});
    weights.push_back(init_weight);
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  //1st. we have to figure out is the target moving straight?
  //if the yaw_rate is close to zero we assume it is going straight
  const bool is_moving_straight = fabs(yaw_rate) < 1e-3;
  const double v_over_yawrate = is_moving_straight ? velocity * delta_t : velocity / yaw_rate;
  const double delta_theta = yaw_rate * delta_t;

  default_random_engine gen;
  normal_distribution<double> noise_x(0, std_pos[0]);
  normal_distribution<double> noise_y(0, std_pos[1]);
  normal_distribution<double> noise_theta(0, std_pos[2]);

  for (auto &particle: particles) {
    //precompute reused values here
    const double sin_theta = sin(particle.theta);
    const double cos_theta = cos(particle.theta);
    const double new_theta = particle.theta + delta_theta;

    //NOTE: This branching sucks especially if I want to run this using GPGPU
    //Perhaps I can just avoid this by adding a small epsilon to yaw_rate if zero
    if (is_moving_straight) {
      //pretty much going straight
      particle.x += v_over_yawrate * cos_theta + noise_x(gen);
      particle.y += v_over_yawrate * sin_theta + noise_y(gen);
      particle.theta += noise_theta(gen);
    } else {
      particle.x += v_over_yawrate * (sin(new_theta) - sin_theta) + noise_x(gen);
      particle.y += v_over_yawrate * (cos_theta - cos(new_theta)) + noise_y(gen);
      particle.theta = new_theta + noise_theta(gen);
    }
  }

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs> &observations) {
  /**
   *  For each observations, we must perform the following
   *
   *  1. Calculate the distance to each possible landmark
   *  2. Select the closest landmark as the ID for the observation
   *     NOTE: I don't need to calculate full L2 Norm, I can save compute by not using sqrt, since
   *     I just need distance!!
   */
  for (auto &observation: observations) {
    auto nearest_landmark_it = min_element(predicted.begin(),
                                           predicted.end(),
                                           [&](const LandmarkObs &left, const LandmarkObs &right) {
                                             //get dist of left and observation
                                             const double left_dx = left.x - observation.x;
                                             const double left_dy = left.y - observation.y;
                                             const double left_distance = pow(left_dx, 2) + pow(left_dy, 2);

                                             const double right_dx = right.x - observation.x;
                                             const double right_dy = right.y - observation.y;
                                             const double right_distance = pow(right_dx, 2) + pow(right_dy, 2);

                                             return left_distance < right_distance;
                                           });

    observation.id = nearest_landmark_it->id;
  }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   std::vector<LandmarkObs> observations, Map map_landmarks) {

  /**
   * For each particle, we must preform the following...
   *
   * 1. Transform observation from car coord system to Map coord with respect to the give particle.
   * 2. Collect Map land_marks within sensor_range as possible landmark.
   * 3. Use dataAssociation method that tags the transformed observation to the closet landmark in range.
   * 4. Use multi-Variate Normal distribution to calculate the likelihood of each particle using observations.
   *    Using importance sampling this gives us a ratio we can use for resampling.
   */

  weights.clear();

  for (auto &particle: particles) {
    //1. Transform observation from car coord system to Map coord with respect to the give particle.
    const double sin_theta = sin(particle.theta);
    const double cos_theta = cos(particle.theta);

    vector<LandmarkObs> map_observations;
    for (const auto &o : observations) {
      map_observations.push_back(
        LandmarkObs {o.id,
                     o.x * cos_theta - o.y * sin_theta + particle.x,
                     o.x * sin_theta + o.y * cos_theta + particle.y
        });
    }

    //2. Collect Map land_marks within sensor_range as possible landmark.
    vector<LandmarkObs> possible_landmarks;
    for (const auto &landmark: map_landmarks.landmark_list) {
      const double distance = dist(landmark.x_f, landmark.y_f, particle.x, particle.y);

      if (distance <= sensor_range) {
        //means this is a possible landmark!
        possible_landmarks.push_back(LandmarkObs {
          static_cast<int>(possible_landmarks.size()),
          landmark.x_f,
          landmark.y_f}
        );
      }
    }

    //3. Use dataAssociation method that tags the transformed observation to the closet landmark in range.
    dataAssociation(possible_landmarks, map_observations);

    //4. Use multi-Variate Normal distribution to calculate the likelihood of each particle using observations.
    //   Using importance sampling this gives us a ratio we can use for resampling.
    const double std_x = std_landmark[0];
    const double std_y = std_landmark[1];

    const double scaler = 1.0 / (2.0 * M_PI * std_x * std_y);
    const double dx_divider = 2.0 * pow(std_x, 2);
    const double dy_divider = 2.0 * pow(std_y, 2);

    double weight = 1;
    for (const auto &map_observation: map_observations) {
      const double dx2 = pow(map_observation.x - possible_landmarks[map_observation.id].x, 2);
      const double dy2 = pow(map_observation.y - possible_landmarks[map_observation.id].y, 2);
      weight *= scaler * exp(-(dx2 / dx_divider + dy2 / dy_divider));

      if (weight == 0)
        break;
    }

    particle.weight = weight;
    weights.push_back(weight);

  } //end particle for
}

void ParticleFilter::resample() {
  default_random_engine gen;
  discrete_distribution<int> probable_particle_index_sampler(weights.begin(), weights.end());

  const double init_weight = 1.0;
  vector<Particle> new_particles;

  for (int i = 0; i < num_particles; ++i) {
    int index = probable_particle_index_sampler(gen);

    new_particles.push_back(Particle {i,
                                      particles[index].x,
                                      particles[index].y,
                                      particles[index].theta,
                                      init_weight});

  }

  particles = new_particles;
}

void ParticleFilter::write(std::string filename) {

  // You don't need to modify this file.
  std::ofstream dataFile;
  dataFile.open(filename, std::ios::app);

  for (int i = 0; i < num_particles; ++i) {
    dataFile << this->particles[i].x << " " << this->particles[i].y << " " << this->particles[i].theta << "\n";
  }

  dataFile.close();
}
