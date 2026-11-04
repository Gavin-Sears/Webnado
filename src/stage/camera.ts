import * as THREE from "three/webgpu";

import { aspectRatio, canvas, device } from "../renderer";

class CameraUniforms {
    // 3 mat4 + cameraPos + frameIndex = 52 values = 208 bytes
    readonly buffer = new ArrayBuffer(208);
    private readonly floatView = new Float32Array(this.buffer);
    private readonly uintView = new Uint32Array(this.buffer);

    set viewProjMat(elements: ArrayLike<number>) {
        this.floatView.set(elements, 0);
    }

    set invViewProjMat(elements: ArrayLike<number>) {
        this.floatView.set(elements, 16);
    }

    set prevViewProjMat(elements: ArrayLike<number>) {
        this.floatView.set(elements, 32);
    }

    setCameraPos(pos: THREE.Vector3) {
        this.floatView[48] = pos.x;
        this.floatView[49] = pos.y;
        this.floatView[50] = pos.z;
    }

    set frameIndex(index: number) {
        this.uintView[51] = index;
    }
}

const MAX_MOVE_STEP_MS = 100;   // a frame gap can last minutes after a hidden tab
const MAX_MOUSE_STEP_PX = 200;  // browsers can report huge jumps when the lock starts
const WORLD_LIMIT_XZ = 30000;   // meters from the center, where the ground ends
const MIN_HEIGHT = 15;          // meters, just above the near plane
const MAX_HEIGHT = 30000;

const MOVEMENT_KEYS = new Set(['KeyW', 'KeyA', 'KeyS', 'KeyD', 'KeyQ', 'KeyE']);
const NON_TEXT_INPUTS = new Set(['button', 'checkbox', 'color', 'radio', 'range']);

// Only elements that take letters keep the movement keys
function isTextEntry(target: EventTarget | null): boolean {
    return (target instanceof HTMLInputElement && !NON_TEXT_INPUTS.has(target.type)) || target instanceof HTMLTextAreaElement
        || (target instanceof HTMLElement && target.isContentEditable);
}

// The Three.js camera is the source of truth, so the scene and raymarcher always agree
export class Camera {
    uniforms = new CameraUniforms();
    uniformsBuffer: GPUBuffer;

    threeCamera = new THREE.PerspectiveCamera(45, 1, 10, 100000);

    yaw = -90;
    pitch = 6;
    moveSpeed = 3; // meters per millisecond
    sensitivity = 0.15;

    private keys = new Set<string>();   // physical keys, so Shift and keyboard layout don't matter
    private sprint = false;
    private looking = false;

    moved = true;   // view changed this frame

    private frameIndex = 0;
    private front = new THREE.Vector3();
    private right = new THREE.Vector3();
    private moveDir = new THREE.Vector3();
    private lookTarget = new THREE.Vector3();
    private viewProjMat = new THREE.Matrix4().set(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    private prevViewProjMat = new THREE.Matrix4();
    private invViewProjMat = new THREE.Matrix4();

    constructor() {
        this.uniformsBuffer = device.createBuffer({
            label: "camera uniforms",
            size: this.uniforms.buffer.byteLength,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        // WebGPU's [0, 1] depth range
        this.threeCamera.coordinateSystem = THREE.WebGPUCoordinateSystem;
        this.threeCamera.updateProjectionMatrix();
        this.threeCamera.position.set(0, 1800, 12000);

        // capture, so a GUI element can't hide a keyup
        window.addEventListener('keydown', (event) => this.onKeyDown(event), true);
        window.addEventListener('keyup', (event) => this.onKeyUp(event), true);

        // releases during any of these are never reported
        window.addEventListener('blur', () => this.releaseInput());
        window.addEventListener('pagehide', () => this.releaseInput());
        window.addEventListener('contextmenu', () => this.releaseInput());
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                this.releaseInput();
            }
        });

        // An open dropdown can swallow key releases, so clicking the GUI ends movement.
        // Capture, so a GUI element can't hide the press.
        window.addEventListener('pointerdown', (event) => {
            if (event.target !== canvas) {
                this.releaseInput();
            }
        }, true);
        window.addEventListener('focusin', () => this.keys.clear());

        canvas.addEventListener('pointerdown', (event) => this.onPointerDown(event));
        canvas.addEventListener('pointermove', (event) => this.onPointerMove(event));
        canvas.addEventListener('pointerup', () => this.endLook());
        canvas.addEventListener('pointercancel', () => this.endLook());
        document.addEventListener('pointerlockchange', () => {
            // released before the lock was granted
            if (document.pointerLockElement === canvas && !this.looking) {
                document.exitPointerLock();
            }
        });
    }

    private onKeyDown(event: KeyboardEvent) {
        this.sprint = event.shiftKey;

        if (!MOVEMENT_KEYS.has(event.code) || event.ctrlKey || event.altKey || event.metaKey || isTextEntry(event.target)) {
            return;
        }

        // a focused dropdown would treat the letter as a search
        if (event.target instanceof HTMLElement && event.target !== document.body) {
            event.preventDefault();
            event.target.blur();
        }

        // a late repeat must not restart movement after a release
        if (!event.repeat) {
            this.keys.add(event.code);
        }
    }

    private onKeyUp(event: KeyboardEvent) {
        this.sprint = event.shiftKey;
        this.keys.delete(event.code);

        // Alt moves focus to the browser menu, and macOS hides key releases while Cmd is held
        if (event.key === 'Alt' || event.key === 'Meta') {
            this.keys.clear();
        }
    }

    private releaseInput() {
        this.keys.clear();
        this.sprint = false;
        this.endLook();
    }

    private onPointerDown(event: PointerEvent) {
        if (event.button !== 0) {
            return;
        }

        this.looking = true;
        window.focus();
        if (document.activeElement instanceof HTMLElement) {
            document.activeElement.blur(); // takes the keys back from a text box
        }
        try {
            canvas.setPointerCapture(event.pointerId); // reports a release outside the window
        } catch {
            // the pointer is gone, so the next move ends the look
        }
        this.lockPointer();
    }

    // The lock only stops the screen edge from limiting the turn. Browsers refuse it in ordinary cases, like a click
    // that focuses the window, and looking works without it.
    private async lockPointer() {
        try {
            await canvas.requestPointerLock();
        } catch {
            // keep looking without the lock
        }
    }

    private endLook() {
        this.looking = false;
        if (document.pointerLockElement === canvas) {
            document.exitPointerLock();
        }
    }

    private onPointerMove(event: PointerEvent) {
        if (!this.looking) {
            return;
        }
        if ((event.buttons & 1) === 0) {
            this.endLook(); // missed the release
            return;
        }

        const { clamp } = THREE.MathUtils;
        this.yaw = (this.yaw + clamp(event.movementX, -MAX_MOUSE_STEP_PX, MAX_MOUSE_STEP_PX) * this.sensitivity) % 360;
        this.pitch = clamp(this.pitch - clamp(event.movementY, -MAX_MOUSE_STEP_PX, MAX_MOUSE_STEP_PX) * this.sensitivity, -89, 89);
    }

    private processInput(deltaTime: number) {
        // focus can leave with no blur event or keyup
        if (this.keys.size > 0 && !document.hasFocus()) {
            this.releaseInput();
        }

        const yaw = THREE.MathUtils.degToRad(this.yaw);
        const pitch = THREE.MathUtils.degToRad(this.pitch);
        this.front.set(Math.cos(yaw) * Math.cos(pitch), Math.sin(pitch), Math.sin(yaw) * Math.cos(pitch));
        this.right.crossVectors(this.front, this.threeCamera.up).normalize();

        this.moveDir.set(0, 0, 0);
        if (this.keys.has('KeyW')) this.moveDir.add(this.front);
        if (this.keys.has('KeyS')) this.moveDir.sub(this.front);
        if (this.keys.has('KeyA')) this.moveDir.sub(this.right);
        if (this.keys.has('KeyD')) this.moveDir.add(this.right);
        if (this.keys.has('KeyQ')) this.moveDir.y -= 1;
        if (this.keys.has('KeyE')) this.moveDir.y += 1;

        const position = this.threeCamera.position;
        if (this.moveDir.lengthSq() > 0) {
            const speedMultiplier = this.sprint ? 3 : 1;
            position.addScaledVector(this.moveDir.normalize(), this.moveSpeed * speedMultiplier * Math.min(deltaTime, MAX_MOVE_STEP_MS));
        }

        position.x = THREE.MathUtils.clamp(position.x, -WORLD_LIMIT_XZ, WORLD_LIMIT_XZ);
        position.y = THREE.MathUtils.clamp(position.y, MIN_HEIGHT, MAX_HEIGHT);
        position.z = THREE.MathUtils.clamp(position.z, -WORLD_LIMIT_XZ, WORLD_LIMIT_XZ);
    }

    onFrame(deltaTime: number) {
        this.processInput(deltaTime);

        const camera = this.threeCamera;
        if (camera.aspect !== aspectRatio) {
            camera.aspect = aspectRatio;
            camera.updateProjectionMatrix();
        }
        camera.lookAt(this.lookTarget.copy(camera.position).add(this.front));
        camera.updateMatrixWorld();

        // zero on the first frame, so any reprojection must treat frameIndex 0 as having no history
        this.prevViewProjMat.copy(this.viewProjMat);
        this.uniforms.prevViewProjMat = this.prevViewProjMat.elements;

        this.viewProjMat.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
        this.moved = !this.viewProjMat.equals(this.prevViewProjMat);
        this.invViewProjMat.copy(this.viewProjMat).invert();

        this.uniforms.viewProjMat = this.viewProjMat.elements;
        this.uniforms.invViewProjMat = this.invViewProjMat.elements;
        this.uniforms.setCameraPos(camera.position);
        this.uniforms.frameIndex = this.frameIndex++;

        device.queue.writeBuffer(this.uniformsBuffer, 0, this.uniforms.buffer);
    }
}
