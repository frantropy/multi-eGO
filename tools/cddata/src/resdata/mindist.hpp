#ifndef _RESDATA_MINDIST_HPP
#define _RESDATA_MINDIST_HPP

#include <cmath>
#include <mutex>
#include <vector>

#include "resdata/indexing.hpp"
#include "resdata/density.hpp"

namespace resdata::mindist
{

static void mindist_same(
  std::size_t start_mti_same, std::size_t start_im_same, std::size_t start_i_same, 
  std::size_t start_j_same, long int n_loop_operations_same, const std::vector<float> &density_bins,
  const std::vector<int> &num_mol_unique, const std::vector<int> &natmol2,
  const std::vector<std::vector<float>> &frame_same_mat,
  std::vector<std::vector<std::mutex>> &frame_same_mutex,
  std::vector<std::vector<std::vector<std::vector<float>>>> &interm_same_maxcdf_mol,
  float weight
)
{
  bool first_im_same = true, first_i_same = true, first_j_same = true;
  unsigned counter_same = 0;
  for (std::size_t mt_i = start_mti_same; mt_i < natmol2.size(); mt_i++)
  {
    for (std::size_t im = (first_im_same) ? start_im_same : 0; im < num_mol_unique[mt_i]; im++)
    {
      for (std::size_t i = (first_i_same) ? start_i_same : 0; i < natmol2[mt_i]; i++)
      {
        for (std::size_t j = (first_j_same) ? start_j_same : i; j < natmol2[mt_i]; j++)
        {
          std::size_t offset = resdata::indexing::offset_same(mt_i, im, i, j, natmol2);
          std::size_t mutex_j = resdata::indexing::mutex_access(mt_i, i, j, natmol2);
          float mindist = frame_same_mat[mt_i][offset];

          std::unique_lock<std::mutex> lock(frame_same_mutex[mt_i][mutex_j]);
          resdata::density::kernel_density_estimator(std::begin(interm_same_maxcdf_mol[mt_i][i][j]), density_bins, mindist, weight);
          interm_same_maxcdf_mol[mt_i][j][i] = interm_same_maxcdf_mol[mt_i][i][j];
          lock.unlock();

          ++counter_same;
          if (counter_same == n_loop_operations_same) return;
        }
        first_j_same = false;
      }
      first_i_same = false;
    }
    first_im_same = false;
  }
}

static void mindist_cross(
  std::size_t start_mti_cross, std::size_t start_mtj_cross, std::size_t start_im_cross, std::size_t start_jm_cross, 
  std::size_t start_i_cross, std::size_t start_j_cross, int n_loop_operations_cross, 
  const std::vector<int> &natmol2, const std::vector<std::vector<int>> &cross_index,
  const std::vector<float> &density_bins, const std::vector<int> &num_mol_unique,
  const std::vector<std::vector<float>> &frame_cross_mat,
  std::vector<std::vector<std::mutex>> &frame_cross_mutex,
  std::vector<std::vector<std::vector<std::vector<float>>>> &interm_cross_maxcdf_mol,
  float weight
)
{
  bool first_im_cross = true, first_mtj_cross = true, first_jm_cross = true, first_i_cross = true, first_j_cross = true;
  unsigned counter_cross = 0;
  
  for (std::size_t mt_i = start_mti_cross; mt_i < natmol2.size(); mt_i++)
  {
    for (std::size_t mt_j = (first_mtj_cross) ? start_mtj_cross : mt_i + 1; mt_j < natmol2.size(); mt_j++)
    {
      for (std::size_t im = (first_im_cross) ? start_im_cross : 0; im < num_mol_unique[mt_i]; im++)
      {
        for (std::size_t jm = (first_jm_cross) ? start_jm_cross : 0; jm < num_mol_unique[mt_j]; jm++)
        {
          for (std::size_t i = (first_i_cross) ? start_i_cross : 0; i < natmol2[mt_i]; i++)
          {
            for (std::size_t j = (first_j_cross) ? start_j_cross : 0; j < natmol2[mt_j]; j++)
            {
              std::size_t offset = resdata::indexing::offset_cross(mt_i, mt_j, im, jm, i, j, natmol2);
              std::size_t mutex_j = resdata::indexing::mutex_access(mt_j, i, j, natmol2);
              float mindist = frame_cross_mat[cross_index[mt_i][mt_j]][offset];
              
              std::unique_lock<std::mutex> lock(frame_cross_mutex[cross_index[mt_i][mt_j]][mutex_j]);
              resdata::density::kernel_density_estimator(std::begin(interm_cross_maxcdf_mol[cross_index[mt_i][mt_j]][i][j]), density_bins, mindist, weight);
              lock.unlock();

              ++counter_cross;
              if (counter_cross == n_loop_operations_cross) return;
            }
            first_j_cross = false;
          }
          first_i_cross = false;
        }
        first_jm_cross = false;
      }
      first_mtj_cross = false;
    }
    first_im_cross = false;
  }
}

static void mindist_kernel( // indices
  const resdata::indexing::SameThreadIndices &same_thread_indices, 
  const resdata::indexing::CrossThreadIndices &cross_thread_indices, 
  const float weight,                            // common parameters
  const std::vector<int> &natmol2,
  const std::vector<float> &density_bins,
  const std::vector<int> &num_mol_unique,
  const std::vector<std::vector<float>> &frame_same_mat,
  std::vector<std::vector<std::mutex>> &frame_same_mutex, 
  std::vector<std::vector<std::vector<std::vector<float>>>> &interm_same_maxcdf_mol,
  const std::vector<std::vector<int>> &cross_index, 
  const std::vector<std::vector<float>> &frame_cross_mat,
  std::vector<std::vector<std::mutex>> &frame_cross_mutex,
  std::vector<std::vector<std::vector<std::vector<float>>>> &interm_cross_maxcdf_mol
)
{
  const std::size_t start_mti_same = same_thread_indices.start_mti_same;
  const std::size_t start_im_same = same_thread_indices.start_im_same;
  const std::size_t start_i_same = same_thread_indices.start_i_same;
  const std::size_t start_j_same = same_thread_indices.start_j_same;
  const long int n_loop_operations_same = same_thread_indices.n_loop_operations_same;

  const std::size_t start_mti_cross = cross_thread_indices.start_mti_cross;
  const std::size_t start_mtj_cross = cross_thread_indices.start_mtj_cross;
  const std::size_t start_im_cross = cross_thread_indices.start_im_cross;
  const std::size_t start_jm_cross = cross_thread_indices.start_jm_cross;
  const std::size_t start_i_cross = cross_thread_indices.start_i_cross;
  const std::size_t start_j_cross = cross_thread_indices.start_j_cross;
  const int n_loop_operations_cross = cross_thread_indices.n_loop_operations_cross;

  if (n_loop_operations_same != 0)
  {
    mindist_same(
      start_mti_same, start_im_same, start_i_same, start_j_same, n_loop_operations_same, density_bins,
      num_mol_unique, natmol2, frame_same_mat, frame_same_mutex, interm_same_maxcdf_mol, weight
    );
  }

  if (n_loop_operations_cross != 0)
  {
    mindist_cross(
      start_mti_cross, start_mtj_cross, start_im_cross, start_jm_cross, start_i_cross, start_j_cross, n_loop_operations_cross, 
      natmol2, cross_index, density_bins, num_mol_unique, frame_cross_mat, frame_cross_mutex, interm_cross_maxcdf_mol, weight
    );
  }
}

} // namespace resdata::mindist

#endif // _RESDATA_MINDIST_HPP