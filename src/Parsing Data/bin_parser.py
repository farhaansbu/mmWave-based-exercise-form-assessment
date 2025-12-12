import os
import csv
import struct
import numpy as np
import matplotlib.pyplot as plt

from parser_mmw_demo import parser_one_mmw_demo_output_packet, getUint32

###############################################################################
# Utility: Extract TLV blocks from a parsed packet
###############################################################################
def extract_tlvs(data, header_start, header_len, num_tlv):
    tlvs = []
    index = header_start + header_len

    for _ in range(num_tlv):
        # Ensure at least 8 bytes remain for TLV header
        if index + 8 > len(data):
            print("WARNING: TLV header truncated — stopping TLV parsing.")
            break

        tlv_type = getUint32(data[index:index+4])
        tlv_len  = getUint32(data[index+4:index+8])

        # Sanity check: TLV length must be positive and sensible
        if tlv_len <= 8:
            print(f"WARNING: Invalid TLV length {tlv_len}, skipping.")
            break

        # Ensure TLV payload is fully present
        if index + tlv_len > len(data):
            print("WARNING: TLV payload truncated — stopping TLV parsing.")
            break

        tlv_payload = data[index+8:index+tlv_len]
        tlvs.append((tlv_type, tlv_payload))

        index += tlv_len
    return tlvs


###############################################################################
# Image reconstruction helpers
###############################################################################
def parse_range_doppler_heatmap(payload, num_range_bins, num_doppler_bins):
    """
    TLV Type 3: Range-Doppler Heatmap
    Values are uint16 (log magnitude).
    """
    values = np.frombuffer(payload, dtype=np.uint16)
    if len(values) != num_range_bins * num_doppler_bins:
        print("WARNING: Unexpected heatmap size")
    return values.reshape((num_doppler_bins, num_range_bins))


def parse_azimuth_static_heatmap(payload, num_range_bins, num_virtual_ant):
    """
    TLV Type 4: Azimuth static heatmap (complex int16)
    Shape: [range_bins, virtual_antennas]
    """
    raw = np.frombuffer(payload, dtype=np.int16)
    complex_vals = raw.reshape((-1, 2))  # [imag, real]
    complex_vals = complex_vals[:, 1] + 1j * complex_vals[:, 0]

    if len(complex_vals) != num_range_bins * num_virtual_ant:
        print("WARNING: Unexpected azimuth heatmap size")

    return complex_vals.reshape((num_range_bins, num_virtual_ant))


def show_heatmap(matrix, title):
    plt.figure(figsize=(6, 4))
    plt.imshow(matrix, aspect='auto')
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.show()


###############################################################################
# Main parser wrapper
###############################################################################
def parse_dat_file(filename,
                   output_csv=None,
                   save_images=True,
                   num_range_bins=256,
                   num_doppler_bins=64,
                   num_virtual_ant=8):
    """
    Complete parser that:
      • Parses frames
      • Saves detected objects to CSV
      • Extracts & plots heatmaps if present
    """

      # Auto-generate CSV name based on input file
    if output_csv is None:
        base = os.path.splitext(os.path.basename(filename))[0]
        output_csv = f"{base}.csv"
    
    print(f"Reading: {filename}")
    with open(filename, "rb") as f:
        bin_data = f.read()
    file_size = len(bin_data)

    print(f"Output CSV: {output_csv}")
    csv_file = open(output_csv, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["frame", "obj_id", "x", "y", "z", "v", "range_m", "azimuth_deg", "elev_deg", "snr", "noise"])

    total_parsed = 0
    frame_idx = 0

    while total_parsed < file_size:
        (result,
         header_start,
         pkt_len,
         num_obj,
         num_tlv,
         subframe,
         x_arr,
         y_arr,
         z_arr,
         v_arr,
         range_arr,
         az_arr,
         elev_arr,
         snr_arr,
         noise_arr) = parser_one_mmw_demo_output_packet(
            bin_data[total_parsed:], file_size - total_parsed
        )

        if result != 0:
            print(f"WARNING: Frame {frame_idx} malformed or empty — adding placeholder row.")

            # Write placeholder frame row (no objects)
            writer.writerow([
                frame_idx, -1,   # frame, obj_id
                0,0,0,0,         # x,y,z,v
                0,0,0,           # range, az, elev
                0,0              # snr, noise
            ])

            # Move to next frame
            total_parsed += (header_start + pkt_len)
            frame_idx += 1
            continue
           

        print(f"Parsed frame {frame_idx}, {num_obj} objects.")

        # Save to CSV
        for i in range(num_obj):
            writer.writerow([
                frame_idx, i,
                x_arr[i], y_arr[i], z_arr[i], v_arr[i],
                range_arr[i], az_arr[i], elev_arr[i],
                snr_arr[i], noise_arr[i]
            ])

        # Extract TLVs for images
        tlvs = extract_tlvs(
            bin_data,
            total_parsed + header_start,
            40,  # TI header size (bytes)
            num_tlv
        )

        for tlv_type, payload in tlvs:
            if tlv_type == 3:  # Range-Doppler heatmap
                rd = parse_range_doppler_heatmap(payload, num_range_bins, num_doppler_bins)
                if save_images:
                    show_heatmap(rd, f"Range-Doppler Heatmap (Frame {frame_idx})")

            elif tlv_type == 4:  # Azimuth static heatmap
                az = parse_azimuth_static_heatmap(payload, num_range_bins, num_virtual_ant)
                if save_images:
                    show_heatmap(np.abs(az), f"Azimuth Heatmap (Frame {frame_idx})")

        total_parsed += (header_start + pkt_len)
        frame_idx += 1

    csv_file.close()
    print(f"Finished. Parsed {frame_idx} frames. CSV saved to {output_csv}")


###############################################################################
# CLI Entry
###############################################################################
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python3 chatgpt_parser.py <datfile>")
        exit(1)

    datfile = sys.argv[1]
    parse_dat_file(datfile)
