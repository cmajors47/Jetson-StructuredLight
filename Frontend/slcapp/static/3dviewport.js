import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

const container = document.getElementById('3dviewport');
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);

const renderer = new THREE.WebGLRenderer();
renderer.setSize(container.clientWidth, container.clientHeight);
container.appendChild(renderer.domElement);

const geometry = new THREE.BoxGeometry( 1, 1, 1 );
const material = new THREE.MeshBasicMaterial({ color: 0x325aa8 });
const cube = new THREE.Mesh(geometry, material);
//scene.add(cube);
// Example lighting setup
const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
scene.add(ambientLight);

const directionalLight = new THREE.DirectionalLight(0xffffff, 0.75);
directionalLight.position.set(1, 1, 1); // Position the light
scene.add(directionalLight);

camera.position.z = 5;

function animate() {
  //cube.rotation.x += 0.01;
  //cube.rotation.y += 0.01;
	renderer.render( scene, camera );
}
renderer.setAnimationLoop( animate );

export function displayModel(filename) {
  const loader = new GLTFLoader();
  loader.load(filename, function(gltf) {
    let model = gltf.scene;
  	scene.add(model);

    model.position.set(0, 0, 0); 
    model.scale.set(100,100,100)
    console.log(model);
    model.traverse((child) => {
      if (child.isMesh) {
        child.material.transparent = false; // Disable transparency
        child.material.opacity = 1; // Set full opacity
      }
    });
    camera.position.set(4, 3, 5);  // Move the camera back
    camera.lookAt(model.position); // Ensure the camera looks at the model
  }, undefined, function(error) {
  	console.error(error);
  });
  return filename
};