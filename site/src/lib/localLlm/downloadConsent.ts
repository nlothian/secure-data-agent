/**
 * Local models the user has agreed to download in this page load: through
 * the size dialog (from the model picker, Settings or a send), or because
 * the model was already cached when checked. A send asks first for a model
 * with no consent here (`useLocalGemmaSwitcher`'s `confirmDownload`), so a
 * multi-GB download never starts unannounced, while a download the user
 * already agreed to (still in progress, so not yet cached) is not asked
 * about again.
 */
const consented = new Set<string>();

export function noteDownloadConsent(modelId: string): void {
  consented.add(modelId);
}

export function hasDownloadConsent(modelId: string): boolean {
  return consented.has(modelId);
}
