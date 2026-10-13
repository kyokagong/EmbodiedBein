export function frameAtTime(timestamps: number[], time: number): number {
  let lo = 0, hi = timestamps.length - 1;
  while (lo < hi) {
    const mid = Math.ceil((lo + hi) / 2);
    if (timestamps[mid] <= time) lo = mid;
    else hi = mid - 1;
  }
  return lo;
}

export function motionPose(data: any, frame: number) {
  const origin = data.root_pos[0];
  const position = data.root_pos[frame];
  return {
    joints: Object.fromEntries(data.joint_order.map((name: string, i: number) => [name, data.q[frame][i]])),
    position: [position[0] - origin[0], position[1] - origin[1], position[2]],
    quaternion: data.root_quat[frame],
  };
}
