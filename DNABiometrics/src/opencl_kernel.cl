typedef struct Node {
  unsigned char bitmap;
  unsigned offset;
  int fail_link;
  int output_link;
  int pattern_idx;
} Node;

__kernel void search_pattern_opencl(
    __global Node *vec, __global char *text, const int text_length,
    __global int *indices, __global unsigned int *sizes,
    __global unsigned int *locks, const int thread_count,
    const int max_pattern_length, const int k, const int max_size) {

  int tid = get_global_id(0);
  int start_index = (text_length / thread_count) * tid;
  int end_index =
      start_index + (text_length / thread_count) + max_pattern_length - 2;

  if (end_index >= text_length) {
    end_index = text_length - 1;
  }

  int cur = 0;
  for (int i = start_index; i <= end_index; i++) {
    int c = text[i] - 'a';
    if (vec[cur].bitmap & (1 << c)) {
      unsigned char maskedNumber = vec[cur].bitmap & ((1 << (c + 1)) - 1);
      cur = vec[cur].offset + popcount(maskedNumber) - 1;

      int overlap =
          ((start_index != 0) && (i - start_index + 1 < max_pattern_length));

      if (overlap) {
        continue;
      }

      if ((vec[cur].pattern_idx >= 0)) {
        int pattern_index = vec[cur].pattern_idx;
        while (atomic_cmpxchg(&locks[pattern_index], 0, 1) != 0)
          ; // Spinlock
        unsigned int idx = atomic_inc(&sizes[pattern_index]);
        indices[pattern_index * max_size + idx] = i;
        atomic_xchg(&locks[pattern_index], 0); // Release lock
      }

      int temp = vec[cur].output_link;
      while (temp != 0) {
        int output_pattern_index = vec[temp].pattern_idx;
        while (atomic_cmpxchg(&locks[output_pattern_index], 0, 1) != 0)
          ; // Spinlock
        unsigned int idx = atomic_inc(&sizes[output_pattern_index]);
        indices[output_pattern_index * max_size + idx] = i;
        atomic_xchg(&locks[output_pattern_index], 0); // Release lock
        temp = vec[temp].output_link;
      }
    } else {
      while (cur != 0 && ((vec[cur].bitmap & (1 << c)) == 0)) {
        cur = vec[cur].fail_link;
      }
      if ((vec[cur].bitmap & (1 << c))) {
        i--;
      }
    }
  }
}
