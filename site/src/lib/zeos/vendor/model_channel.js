// Vendored from ZEOS demo/coop-count-web/web/model_channel.js
// at 7149bfc5e8f7fdc96a0ad7b8cce22c204a4a38f5 by site/scripts/zeos-sync.mjs.
// Do not edit here; change it in ZEOS and re-run `npm run zeos:sync`.
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Metacognition AI
//
// This source code is licensed under the AGPL-3.0-only licence found in the
// LICENSE file in the root directory of this source tree.

/**
 * The synchronous `ZeosModelWorker` the Python side calls, over a model that answers
 * asynchronously on another thread.
 *
 * `JsMachine.decode` is synchronous, the kernel loop that calls it is synchronous and
 * stays so, and ONNX Runtime's `session.run` returns a promise. The two meet here: the
 * model runs in its own thread (a Web Worker in the page, a `worker_threads` Worker under
 * Node), and `SyncModelWorker`, on the thread running Pyodide, posts each call to it and
 * then blocks in `Atomics.wait` on a SharedArrayBuffer until the reply has been written
 * into that buffer. The model thread is never blocked, so it receives the message, awaits
 * the session and notifies. `Atomics.wait` is allowed in a Web Worker and on Node's main
 * thread but not on a browser's main thread, which is why Pyodide runs in a worker in the
 * page; SharedArrayBuffer needs a cross-origin isolated page (see `coi.js`).
 *
 * Layout of the buffer: int32 slot 1 is the reply's length in bytes and slot 2 the id of
 * the request it answers; the reply frame (`frames.js`) starts at byte 16. The caller
 * waits for slot 2 to name its own request, not merely to change: a call that timed out
 * is still answered later, and that late reply must not be taken as the next call's.
 * The model thread answers requests in the order they came, so once a call's own reply
 * is there, every earlier one has been written and nothing overwrites it before the
 * caller sends again.
 */

import { decodeFrame, encodeFrame, serveRequest } from "./frames.js";

const LENGTH = 1;
const ANSWERED = 2;
const DATA = 16;
export const CHANNEL_BYTES = 32 << 20;

/** Model side: answer every request arriving through `onMessage` into `buffer`. */
export function serveChannel(worker, buffer, onMessage) {
  const state = new Int32Array(buffer, 0, 4);
  const bytes = new Uint8Array(buffer);
  let chain = Promise.resolve();
  onMessage((request) => {
    chain = chain.then(async () => {
      let frame = encodeFrame(await serveRequest(worker, request));
      if (DATA + frame.byteLength > buffer.byteLength) {
        frame = encodeFrame({
          id: request.id,
          ok: false,
          error: `RangeError: a ${frame.byteLength}-byte reply does not fit the channel`,
        });
      }
      bytes.set(frame, DATA);
      Atomics.store(state, LENGTH, frame.byteLength);
      Atomics.store(state, ANSWERED, request.id);
      Atomics.notify(state, ANSWERED);
    });
  });
}

/** Python side: the `ZeosModelWorker` interface, every method synchronous. */
export class SyncModelWorker {
  /**
   * @param {SharedArrayBuffer} buffer the buffer the model thread answers into.
   * @param {(message: object) => void} post sends a request to the model thread.
   * @param {object} [options]
   * @param {number} [options.timeoutMs] how long one call may take before it throws.
   */
  constructor(buffer, post, { timeoutMs = 600_000 } = {}) {
    this.state = new Int32Array(buffer, 0, 4);
    this.bytes = new Uint8Array(buffer);
    this.post = post;
    this.timeoutMs = timeoutMs;
    // Never the id slot 2 already holds, or the first call would take that reply as
    // its own.
    this.next = (Atomics.load(this.state, ANSWERED) + 1) | 0;
    this.cachedInfo = null;
    // `JsMachine` asks for every piece once at start-up; one call fetches them all.
    this.pieces = this.call("pieces");
    this.backend = this.call("backend");
  }

  call(method, ...args) {
    const id = this.next;
    this.next = (id + 1) | 0;
    this.post({ id, method, args });
    const deadline = Date.now() + this.timeoutMs;
    for (;;) {
      const answered = Atomics.load(this.state, ANSWERED);
      if (answered === id) break;
      const left = deadline - Date.now();
      if (left <= 0) {
        throw new Error(`model worker did not answer ${method} within ${this.timeoutMs} ms`);
      }
      Atomics.wait(this.state, ANSWERED, answered, left);
    }
    const length = Atomics.load(this.state, LENGTH);
    const reply = decodeFrame(this.bytes.slice(DATA, DATA + length));
    if (reply.id !== id) throw new Error(`model worker answered request ${reply.id} as ${id}`);
    if (!reply.ok) throw new Error(`model worker ${method}: ${reply.error}`);
    return reply.value;
  }

  info() {
    if (this.cachedInfo === null) this.cachedInfo = this.call("info");
    return this.cachedInfo;
  }

  tokenize(text) {
    return this.call("tokenize", text);
  }

  piece(tokenId) {
    if (!(tokenId >= 0 && tokenId < this.pieces.length)) {
      throw new RangeError(`token id ${tokenId} is outside the vocabulary`);
    }
    return this.pieces[tokenId];
  }

  /** Fetched for every partial piece at once on the first call; see `frames.js`. */
  pieceBytes(tokenId) {
    if (this.partial === undefined) this.partial = new Map(this.call("partialPieces"));
    const bytes = this.partial.get(tokenId);
    if (bytes === undefined) throw new RangeError(`token id ${tokenId} is whole characters`);
    return bytes;
  }

  createContext(jobId) {
    this.call("createContext", jobId);
  }

  destroyContext(jobId) {
    this.call("destroyContext", jobId);
  }

  length(jobId) {
    return this.call("length", jobId);
  }

  append(jobId, ids) {
    this.call("append", jobId, Int32Array.from(ids));
  }

  truncate(jobId, n) {
    this.call("truncate", jobId, n);
  }

  fork(parentId, childId) {
    this.call("fork", parentId, childId);
  }

  decodeStep(jobId, opts) {
    const { allowedBlocks = null, allowedTokens = null, sample = null } = opts ?? {};
    const args = {
      allowedBlocks: allowedBlocks === null ? null : Uint8Array.from(allowedBlocks),
      allowedTokens: allowedTokens === null ? null : Uint8Array.from(allowedTokens),
    };
    if (sample !== null) args.sample = { temperature: sample.temperature, topK: sample.topK, u: sample.u };
    return this.call("decodeStep", jobId, args);
  }
}
