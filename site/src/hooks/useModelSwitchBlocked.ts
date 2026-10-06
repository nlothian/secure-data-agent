import { useSyncExternalStore } from 'react';
import {
  getModelSwitchServerSnapshot,
  getModelSwitchSnapshot,
  subscribeModelSwitch,
} from '../lib/localLlm/engineLifecycle';

/** Why the chat model cannot be switched now (a reply is streaming), or null. */
export default function useModelSwitchBlocked(): string | null {
  return useSyncExternalStore(
    subscribeModelSwitch,
    getModelSwitchSnapshot,
    getModelSwitchServerSnapshot,
  );
}
