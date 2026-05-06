#include "crypto.h"

#include <stddef.h>
#include <stdint.h>
#include <tee_internal_api.h>

static uint32_t key_len_to_bits(size_t key_len)
{
    if (key_len == TEEGNN_AES128_KEY_LEN) {
        return 128;
    }
    if (key_len == TEEGNN_AES256_KEY_LEN) {
        return 256;
    }
    return 0;
}

static teegnn_status_t tee_to_teegnn_status(TEE_Result res)
{
    switch (res) {
    case TEE_SUCCESS:
        return TEEGNN_OK;
    case TEE_ERROR_BAD_PARAMETERS:
        return TEEGNN_ERR_INVALID_ARG;
    case TEE_ERROR_OUT_OF_MEMORY:
        return TEEGNN_ERR_ALLOC;
    case TEE_ERROR_MAC_INVALID:
        return TEEGNN_ERR_AUTH_FAIL;
    default:
        return TEEGNN_ERR_CRYPTO;
    }
}

static teegnn_status_t validate_common_args(
    const uint8_t *key,
    size_t key_len,
    const uint8_t nonce[TEEGNN_GCM_NONCE_LEN],
    const uint8_t *aad,
    size_t aad_len,
    const uint8_t *input,
    size_t input_len,
    const uint8_t *tag,
    uint8_t *output
)
{
    if (key == NULL || nonce == NULL || tag == NULL) {
        return TEEGNN_ERR_INVALID_ARG;
    }

    if (key_len_to_bits(key_len) == 0) {
        return TEEGNN_ERR_INVALID_ARG;
    }

    if (aad_len > 0 && aad == NULL) {
        return TEEGNN_ERR_INVALID_ARG;
    }

    if (input_len > 0 && (input == NULL || output == NULL)) {
        return TEEGNN_ERR_INVALID_ARG;
    }

    if (aad_len > UINT32_MAX || input_len > UINT32_MAX) {
        return TEEGNN_ERR_INVALID_ARG;
    }

    return TEEGNN_OK;
}

static teegnn_status_t prepare_aes_gcm_operation(
    const uint8_t *key,
    size_t key_len,
    uint32_t mode,
    TEE_OperationHandle *op_out,
    TEE_ObjectHandle *key_obj_out
)
{
    TEE_Result res;
    TEE_Attribute attr;
    TEE_OperationHandle op = TEE_HANDLE_NULL;
    TEE_ObjectHandle key_obj = TEE_HANDLE_NULL;
    uint32_t key_bits = key_len_to_bits(key_len);

    if (key_bits == 0 || op_out == NULL || key_obj_out == NULL) {
        return TEEGNN_ERR_INVALID_ARG;
    }

    res = TEE_AllocateTransientObject(TEE_TYPE_AES, key_bits, &key_obj);
    if (res != TEE_SUCCESS) {
        return tee_to_teegnn_status(res);
    }

    TEE_InitRefAttribute(&attr, TEE_ATTR_SECRET_VALUE, key, key_len);

    res = TEE_PopulateTransientObject(key_obj, &attr, 1);
    if (res != TEE_SUCCESS) {
        TEE_FreeTransientObject(key_obj);
        return tee_to_teegnn_status(res);
    }

    res = TEE_AllocateOperation(&op, TEE_ALG_AES_GCM, mode, key_bits);
    if (res != TEE_SUCCESS) {
        TEE_FreeTransientObject(key_obj);
        return tee_to_teegnn_status(res);
    }

    res = TEE_SetOperationKey(op, key_obj);
    if (res != TEE_SUCCESS) {
        TEE_FreeOperation(op);
        TEE_FreeTransientObject(key_obj);
        return tee_to_teegnn_status(res);
    }

    *op_out = op;
    *key_obj_out = key_obj;
    return TEEGNN_OK;
}

teegnn_status_t teegnn_aes_gcm_encrypt(
    const uint8_t *key,
    size_t key_len,
    const uint8_t nonce[TEEGNN_GCM_NONCE_LEN],
    const uint8_t *aad,
    size_t aad_len,
    const uint8_t *plaintext,
    size_t plaintext_len,
    uint8_t *ciphertext,
    uint8_t tag[TEEGNN_GCM_TAG_LEN]
)
{
    teegnn_status_t st;
    TEE_Result res;
    TEE_OperationHandle op = TEE_HANDLE_NULL;
    TEE_ObjectHandle key_obj = TEE_HANDLE_NULL;
    size_t out_len = plaintext_len;
    size_t tag_len = TEEGNN_GCM_TAG_LEN;
    uint8_t dummy = 0;

    st = validate_common_args(
        key, key_len, nonce, aad, aad_len,
        plaintext, plaintext_len, tag, ciphertext
    );
    if (st != TEEGNN_OK) {
        return st;
    }

    st = prepare_aes_gcm_operation(
        key, key_len, TEE_MODE_ENCRYPT, &op, &key_obj
    );
    if (st != TEEGNN_OK) {
        return st;
    }

    TEE_AEInit(
        op,
        nonce,
        TEEGNN_GCM_NONCE_LEN,
        TEEGNN_GCM_TAG_LEN * 8U,
        (uint32_t)aad_len,
        (uint32_t)plaintext_len
    );

    if (aad_len > 0) {
        TEE_AEUpdateAAD(op, aad, aad_len);
    }

    res = TEE_AEEncryptFinal(
        op,
        plaintext_len > 0 ? plaintext : &dummy,
        plaintext_len,
        plaintext_len > 0 ? ciphertext : NULL,
        &out_len,
        tag,
        &tag_len
    );

    if (res != TEE_SUCCESS) {
        st = tee_to_teegnn_status(res);
        goto out;
    }

    if (out_len != plaintext_len || tag_len != TEEGNN_GCM_TAG_LEN) {
        st = TEEGNN_ERR_CRYPTO;
        goto out;
    }

    st = TEEGNN_OK;

out:
    if (op != TEE_HANDLE_NULL) {
        TEE_FreeOperation(op);
    }
    if (key_obj != TEE_HANDLE_NULL) {
        TEE_FreeTransientObject(key_obj);
    }
    return st;
}

teegnn_status_t teegnn_aes_gcm_decrypt(
    const uint8_t *key,
    size_t key_len,
    const uint8_t nonce[TEEGNN_GCM_NONCE_LEN],
    const uint8_t *aad,
    size_t aad_len,
    const uint8_t *ciphertext,
    size_t ciphertext_len,
    const uint8_t tag[TEEGNN_GCM_TAG_LEN],
    uint8_t *plaintext
)
{
    teegnn_status_t st;
    TEE_Result res;
    TEE_OperationHandle op = TEE_HANDLE_NULL;
    TEE_ObjectHandle key_obj = TEE_HANDLE_NULL;
    size_t out_len = ciphertext_len;
    uint8_t dummy = 0;
    uint8_t tag_buf[TEEGNN_GCM_TAG_LEN];

    st = validate_common_args(
        key, key_len, nonce, aad, aad_len,
        ciphertext, ciphertext_len, tag, plaintext
    );
    if (st != TEEGNN_OK) {
        return st;
    }

    st = prepare_aes_gcm_operation(
        key, key_len, TEE_MODE_DECRYPT, &op, &key_obj
    );
    if (st != TEEGNN_OK) {
        return st;
    }

    TEE_AEInit(
        op,
        nonce,
        TEEGNN_GCM_NONCE_LEN,
        TEEGNN_GCM_TAG_LEN * 8U,
        (uint32_t)aad_len,
        (uint32_t)ciphertext_len
    );

    if (aad_len > 0) {
        TEE_AEUpdateAAD(op, aad, aad_len);
    }

    TEE_MemMove(tag_buf, tag, sizeof(tag_buf));

    res = TEE_AEDecryptFinal(
        op,
        ciphertext_len > 0 ? ciphertext : &dummy,
        ciphertext_len,
        ciphertext_len > 0 ? plaintext : NULL,
        &out_len,
        tag_buf,
        TEEGNN_GCM_TAG_LEN
    );

    if (res != TEE_SUCCESS) {
        st = tee_to_teegnn_status(res);
        goto out;
    }

    if (out_len != ciphertext_len) {
        st = TEEGNN_ERR_CRYPTO;
        goto out;
    }

    st = TEEGNN_OK;

out:
    if (op != TEE_HANDLE_NULL) {
        TEE_FreeOperation(op);
    }
    if (key_obj != TEE_HANDLE_NULL) {
        TEE_FreeTransientObject(key_obj);
    }
    return st;
}
