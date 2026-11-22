/*
 * Copyright 2026 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package io.github.inference4j.vision;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

/** Expected pixels come from {@code cv2.resize(..., interpolation=cv2.INTER_CUBIC)}, OpenCV 5.1. */
class PaddleImagesTest {

    @Test
    void cubicEnlargementMatchesOpenCv() {
        int[] src = {
                0x8B4AE5, 0xF1A941, 0x06A095, 0x6A26AF, 0xBCCDAF,
                0xE562F9, 0x0A945F, 0x5693C6, 0x42276A, 0xD5AB2D,
                0xA73945, 0x51370E, 0x99B2D7, 0x4C3427, 0xFA48D7,
                0x32A1DF, 0x70ACA1, 0xE92611, 0x5901DD, 0x06F27F};
        int[] expected = {
                0x773FF2, 0xD07396, 0xFDAF3C, 0x46B76C, 0x0073A4, 0x6425B3, 0xA27DBB, 0xC1E0BB,
                0xC453FF, 0xA67CB1, 0x70AA55, 0x36AC8E, 0x276CB1, 0x4E2496, 0x997773, 0xCFD459,
                0xFB59EB, 0x7B6B98, 0x008654, 0x40A0B2, 0x5D75BC, 0x3B2B5F, 0x985F3E, 0xF0A139,
                0xC73658, 0x802D23, 0x3F4116, 0x75A3BA, 0x75A0BC, 0x443D23, 0xB2356D, 0xFF48D3,
                0x676986, 0x5F6969, 0x6C6950, 0xB87667, 0xAE5178, 0x571B79, 0x6559A0, 0x889FC1,
                0x21A9F2, 0x3EBCDF, 0x7CAFA2, 0xE04B15, 0xD70038, 0x6600E5, 0x128EBD, 0x00FF6C};

        assertThat(PaddleImages.resizeCubic(src, 5, 4, 8, 6)).containsExactly(expected);
    }

    @Test
    void cubicReductionMatchesOpenCv() {
        int[] src = {
                0x8F063C, 0xD25E19, 0xA621F9, 0xBD0CCC, 0x21397C, 0x1EC795, 0xCA7748, 0xDC03D1,
                0x7A8893, 0x4D8552, 0x7357A2, 0xE64647, 0xF03E2F, 0xB81F22, 0x3F4192, 0xC58EFD,
                0x5185F0, 0x711EF7, 0x677A1F, 0x132A81, 0x8D8821, 0x95A100, 0xB28D6B, 0x236B82,
                0x481BD9, 0xFE630D, 0xC3CE3B, 0xE4EBCA, 0xB72F45, 0xB35223, 0x479F6E, 0x794D57,
                0x037E2C, 0xFDE9E1, 0x5A7789, 0x781E37, 0x21A557, 0xD8E5A7, 0x03296D, 0xCB81DB,
                0x3F65CF, 0x9C1BDC, 0x5D1D8F, 0xC83F0B, 0x0BB4DB, 0x71BE57, 0x220985, 0x1D3D24};
        int[] expected = {
                0xC25C0F, 0x7E1794, 0xCF5E5C,
                0x457199, 0xD14931, 0x545CA6,
                0xAE2399, 0x897488, 0x80A06F,
                0xF7D193, 0x69624F, 0x174B75,
                0x9B36E0, 0x65786D, 0x19037B};

        assertThat(PaddleImages.resizeCubic(src, 8, 6, 3, 5)).containsExactly(expected);
    }
}
