#include "../include/base_realsense_node.h"
#include "librealsense2/rsutil.h"

#include "thrust/device_vector.h"

struct __attribute__((__packed__)) ui8RGB {
    uint8_t r = 0;
    uint8_t g = 0;
    uint8_t b = 0;
};

struct PointXYZ {
  float x = 0;
  float y = 0;
  float z = 0;
  float _pad;

  /// Empty Constructor for initial declarations
  __host__ __device__
  inline PointXYZ(){};

  /// Constructor from values
  __host__ __device__
  inline PointXYZ(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {};
};

/**
 * Point Struct
 * Represents required structure from PointXYZRGB PointCloud ROS Messages
 * Including Padding of the two parts rgb and xyz into two WORDS
 */
struct PointXYZRGB {
  float x = 0;
  float y = 0;
  float z = 0;
  float _pad0;
  float rgb = 0;
  float _pad_1;
  float _pad_2;
  float _pad_3;

  /// Empty Constructor for initial declarations
  __host__ __device__
  inline PointXYZRGB(){};

  /// Constructor from values
  __host__ __device__
  inline PointXYZRGB(float x_, float y_, float z_, float rgb_) 
    : x(x_), y(y_), z(z_), rgb(rgb_) {};

  /// Constructor from individual rgb
  __host__ __device__
  inline PointXYZRGB(float x_, float y_, float z_, uint8_t r, uint8_t g, uint8_t b) : x(x_), y(y_), z(z_)
  {
    uint8_t* rgb_ = (uint8_t*) &rgb;
    rgb_[0] = r;
    rgb_[1] = g;
    rgb_[2] = b;
    rgb_[3] = 0;
  };
};


/**
 * Calculates the Index in the Texture Buffer from the Texture Coordinate
 */
struct CoordinateToIndex {

    int width;
    int height;
    int bytes_per_pixel;
    int stride_in_bytes;

    CoordinateToIndex(rs2::video_frame h_colors) {
        // Get Width and Height coordinates of texture
        width  = h_colors.get_width();  // Frame width in pixels
        height = h_colors.get_height(); // Frame height in pixels
        bytes_per_pixel = h_colors.get_bytes_per_pixel();
        stride_in_bytes = h_colors.get_stride_in_bytes();
    };

    __host__ __device__ 
    int operator()(rs2::texture_coordinate Texture_XY) {       
        // Normals to Texture Coordinates conversion
        int x_value = int(Texture_XY.u * width  + .5f);
        int y_value = int(Texture_XY.v * height + .5f);

        if(x_value < 0 || x_value > width - 1)  return -1;
        if(y_value < 0 || y_value > height - 1) return -1;

        int bytes = x_value;   // Get # of bytes per pixel
        int strides = y_value * width; // Get line width in bytes
        int text_index =  (bytes + strides);

        return text_index;
    }
};


/**
 * Produces a Point as Voxel from a RS Vertex and its RGB
 */
struct MakePointXYZRGB {
    __host__ __device__ 
    PointXYZRGB operator()(rs2::vertex v, ui8RGB c) {
        return PointXYZRGB(v.x, v.y, v.z, c.r, c.g, c.b);
    }
};

struct MakePointXYZ {
    __host__ __device__ 
    PointXYZ operator()(rs2::vertex v) {
        return PointXYZ(v.x, v.y, v.z);
    }
};


struct is_positive
{
  __host__ __device__
  bool operator()(const int &x)
  {
    return x >= 0;
  }
};

struct is_negative
{
  __host__ __device__
  bool operator()(const int &x) { return x < 0; }
};

struct is_negative_xyz
{
  __host__ __device__
  bool operator()(const PointXYZ &p)    { return p.z < 0; }
};

struct is_negative_xyzrgb
{
  __host__ __device__
  bool operator()(const PointXYZRGB &p) { return p.z < 0; }
};




/*************************
 * Functions for RS Data *
 *************************/

/**
 * Parses data given by the Realsense to a PointCloud
 * @param h_points: Points usually create from a depth_frame
 * @param colors: Video Frame for RGB Information
 * @param d_out: Cloud used to save finished Voxels
 */
int cudaFillColorPointCloud(const rs2::points& h_points, const rs2::video_frame& colors, sensor_msgs::PointCloud2Ptr& point_cloud, bool _ordered_pc, bool _allow_no_texture_points){
    PointXYZRGB* h_out = reinterpret_cast<PointXYZRGB*>(point_cloud->data.data());

    const ui8RGB* h_colors = reinterpret_cast<const ui8RGB *>(colors.get_data());
    const rs2::vertex* h_vertex = h_points.get_vertices();
    const rs2::texture_coordinate* h_texture_coord = h_points.get_texture_coordinates();
    
    const size_t point_count = h_points.size();

    thrust::device_vector<PointXYZRGB> d_out(point_count);
    thrust::device_vector<rs2::vertex> d_vertex(point_count);
    thrust::device_vector<ui8RGB>      d_colors(colors.get_height() * colors.get_width());
    thrust::device_vector<int>         d_color_idxs(point_count);
    thrust::device_vector<rs2::texture_coordinate> d_texture_coord(point_count);

    thrust::copy(h_vertex, h_vertex + point_count, d_vertex.begin());
    thrust::copy(h_colors, h_colors + d_colors.size(), d_colors.begin());
    thrust::copy(h_texture_coord, h_texture_coord + point_count, d_texture_coord.begin());

    // Calculate all the index for all texture coordinates...
    thrust::transform(d_texture_coord.begin(), d_texture_coord.end(), d_color_idxs.begin(), CoordinateToIndex(colors));

    auto i_texture_begin = thrust::make_permutation_iterator(d_colors.begin(), d_color_idxs.begin());

    int new_size;
    auto zero_rgb = thrust::make_constant_iterator(ui8RGB());
    if(_ordered_pc) {
        // Produce all the Points
        thrust::transform_if(d_vertex.begin(), d_vertex.end(), i_texture_begin, d_color_idxs.begin(), d_out.begin(), MakePointXYZRGB(), is_positive());
        thrust::transform_if(d_vertex.begin(), d_vertex.end(), zero_rgb,        d_color_idxs.begin(), d_out.begin(), MakePointXYZRGB(), is_negative());
    } else {
        // Produce all the Points
        thrust::transform_if(d_vertex.begin(), d_vertex.end(), i_texture_begin, d_color_idxs.begin(), d_out.begin(), MakePointXYZRGB(), is_positive());
        thrust::transform_if(d_vertex.begin(), d_vertex.end(), zero_rgb,        d_color_idxs.begin(), d_out.begin(), MakePointXYZRGB(), is_negative());

        // Remove all color idxs with invalid vertices
        new_size = thrust::remove_if(d_color_idxs.begin(), d_color_idxs.end(), d_out.begin(), is_negative_xyzrgb()) - d_color_idxs.begin();
        
        // Remove all invalid vertices
        thrust::remove_if(d_out.begin(), d_out.end(), is_negative_xyzrgb());

        if(_allow_no_texture_points) {
            // Do not remove Points without color
            d_color_idxs.resize(new_size);
            d_out.resize(new_size);
        } else {
            // Do remove points without color
            new_size = thrust::remove_if(d_out.begin(), d_out.begin() + new_size, d_color_idxs.begin(), is_negative()) - d_out.begin();
            d_out.resize(new_size);
        }
    }

    thrust::copy(d_out.begin(), d_out.end(), h_out); 

    return new_size;
}

/**
 * Parses data given by the Realsense to a PointCloud
 * @param h_points: Points usually create from a depth_frame
 * @param point_cloud: Cloud used to save finished Voxels
 */
int cudaFillPointCloud(const rs2::points& h_points, sensor_msgs::PointCloud2Ptr& point_cloud, bool _ordered_pc, bool _allow_no_texture_points){
    PointXYZ* h_out = reinterpret_cast<PointXYZ *>(point_cloud->data.data());

    const rs2::vertex* h_vertex = h_points.get_vertices();
    
    const size_t point_count = h_points.size();
    
    thrust::device_vector<PointXYZ> d_out(point_count);
    thrust::device_vector<rs2::vertex> d_vertex(point_count);

    thrust::copy(h_vertex, h_vertex + point_count, d_vertex.begin());

    int new_size = point_count;
    // Produce all the Points
    thrust::transform(d_vertex.begin(), d_vertex.end(), d_out.begin(), MakePointXYZ());
    if(!_ordered_pc) {
        new_size = thrust::remove_if(d_out.begin(), d_out.end(), is_negative_xyz()) - d_out.begin();
    }
    thrust::copy(d_out.begin(), d_out.end(), h_out);
    return new_size;
}
