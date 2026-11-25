// Keeps BoardCheck's inference threads on the big cores (found with Arm Streamline: 27-42 % of the
// inference cycles ran on the small Cortex-A510 cores, at about a third of the A715's work per cycle).
// pin(true): every thread of this process may only run on the fastest cores; pin(false): all cores again.
// Uses sched_setaffinity on our own threads only (no root needed).
#define _GNU_SOURCE
#include <jni.h>
#include <sched.h>
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

static long max_freq(int cpu) {
  char p[96]; snprintf(p, sizeof p, "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", cpu);
  FILE *f = fopen(p, "r"); long v = 0; if (f) { if (fscanf(f, "%ld", &v) != 1) v = 0; fclose(f); } return v;
}

JNIEXPORT jint JNICALL Java_com_boardcheck_app_Affinity_nPin(JNIEnv *e, jclass c, jboolean big) {
  int n = (int) sysconf(_SC_NPROCESSORS_CONF); long top = 0;
  for (int i = 0; i < n; i++) { long f = max_freq(i); if (f > top) top = f; }
  cpu_set_t set; CPU_ZERO(&set); int cores = 0;
  for (int i = 0; i < n; i++) if (!big || top == 0 || max_freq(i) == top) { CPU_SET(i, &set); cores++; }
  DIR *d = opendir("/proc/self/task"); struct dirent *t; int ok = 0;
  if (!d) return -1;
  while ((t = readdir(d))) { int tid = atoi(t->d_name); if (tid > 0 && sched_setaffinity(tid, sizeof set, &set) == 0) ok++; }
  closedir(d);
  return big ? cores : 0;   // cores allowed (0 = unpinned)
}
