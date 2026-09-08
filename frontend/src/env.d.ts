declare module "*.vue" {
  import type { DefineComponent } from "vue";
  const c: DefineComponent<{}, {}, any>;
  export default c;
}

declare module "three";
declare module "urdf-loader";
declare module "three/examples/jsm/controls/OrbitControls.js";
declare module "three/examples/jsm/loaders/STLLoader.js";
