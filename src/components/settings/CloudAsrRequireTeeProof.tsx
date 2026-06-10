import React from "react";
import { useTranslation } from "react-i18next";
import { ToggleSwitch } from "../ui/ToggleSwitch";
import { useSettings } from "../../hooks/useSettings";

interface CloudAsrRequireTeeProofProps {
  descriptionMode?: "inline" | "tooltip";
  grouped?: boolean;
}

/**
 * Strict mode: reject cloud ASR responses that do not include a
 * `tee_proof` field, meaning the provider could not attest the
 * computation ran inside a TEE.
 *
 * Off by default because the 0G router may also forward to non-TeeML
 * providers during the rollout phase; flipping this on guarantees that
 * raw audio only ever decrypts inside a hardware enclave the operator
 * cannot inspect.
 */
export const CloudAsrRequireTeeProof: React.FC<CloudAsrRequireTeeProofProps> =
  React.memo(({ descriptionMode = "tooltip", grouped = false }) => {
    const { t } = useTranslation();
    const { getSetting, updateSetting, isUpdating } = useSettings();

    const required = getSetting("cloud_asr_require_tee_proof") || false;
    const cloudEnabled = getSetting("cloud_asr_enabled") || false;

    return (
      <ToggleSwitch
        checked={required}
        onChange={(v) => updateSetting("cloud_asr_require_tee_proof", v)}
        isUpdating={isUpdating("cloud_asr_require_tee_proof")}
        label={t("settings.cloudAsr.requireTeeProof.label")}
        description={t("settings.cloudAsr.requireTeeProof.description")}
        descriptionMode={descriptionMode}
        grouped={grouped}
        disabled={!cloudEnabled}
      />
    );
  });
