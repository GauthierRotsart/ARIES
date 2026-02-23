import numpy as np


def create_rgb_image(model, mask, border, com_voxel, axis):
    if axis == 'YZ':
        gtv_slice = com_voxel[0]
        rotated_border = np.rot90(border[gtv_slice, :, :])
        img = np.rot90(model.midp.imageArray[gtv_slice, :, :])
        mask = np.rot90(mask.imageArray[gtv_slice, :, :])
    elif axis == 'XZ':
        gtv_slice = com_voxel[1]
        rotated_border = np.rot90(border[:, gtv_slice, :])
        img = np.rot90(model.midp.imageArray[:, gtv_slice, :])
        mask = np.rot90(mask.imageArray[:, gtv_slice, :])
    elif axis == 'XY':
        gtv_slice = com_voxel[2]
        rotated_border = np.rot90(border[:, :, gtv_slice])
        img = np.rot90(model.midp.imageArray[:, :, gtv_slice])
        mask = np.rot90(mask.imageArray[:, :, gtv_slice])
    else:
        raise ValueError

    alpha_channel = (rotated_border * 255).astype(np.uint8)

    # Créer une image RGB rouge
    height, width = alpha_channel.shape
    rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
    rgb_image[..., 0] = 255  # Rendre l'image rouge (canal R)
    rgb_image[..., 1] = 0  # Canal vert
    rgb_image[..., 2] = 0  # Canal bleu

    # Ajouter le canal alpha
    border = np.dstack((rgb_image, alpha_channel))
    return img, mask, border


def plot_function(fig, ax, fixed_model, fixed_mask, fixed_border, fixed_com, registered_model, registered_mask,
                  registered_border, registered_com, axis):
    img_fixed, mask_fixed, border_fixed = create_rgb_image(model=fixed_model,
                                                           mask=fixed_mask,
                                                           border=fixed_border,
                                                           com_voxel=fixed_com,
                                                           axis=axis)
    img_registered, mask_registered, border_registered = create_rgb_image(model=registered_model,
                                                                          mask=registered_mask,
                                                                          border=registered_border,
                                                                          com_voxel=registered_com,
                                                                          axis=axis)
    if axis == 'YZ':
        coord_u = 1
        coord_v = 2
    elif axis == 'XZ':
        coord_u = 0
        coord_v = 2
    elif axis == 'XY':
        coord_u = 0
        coord_v = 1
    else:
        raise ValueError
    coord_x_fixed = fixed_com[coord_u]
    coord_y_fixed = fixed_model.midp.imageArray.shape[coord_v] - fixed_com[coord_v]
    coord_x_registered = registered_com[coord_u]
    coord_y_registered = registered_model.midp.imageArray.shape[coord_v] - registered_com[coord_v]

    # Taille et couleur du marqueur
    size = 100
    color = 'red'

    # Fixed model (T1)
    ax[0, 0].imshow(img_fixed)
    ax[0, 0].set_title('Fixed model')
    ax[0, 0].imshow(border_fixed)
    ax[0, 0].scatter(coord_x_fixed, coord_y_fixed, marker="x", s=size, color=color)

    # Moving model (T2) registered
    ax[0, 1].imshow(img_registered)
    ax[0, 1].set_title('Moving model after deformation')
    ax[0, 1].imshow(border_registered)
    ax[0, 1].scatter(coord_x_registered, coord_y_registered, marker="x", s=size, color=color)

    ax[1, 0].imshow(img_fixed - img_registered)
    ax[1, 0].set_title('Difference fixed - moving after RR')

    ax[1, 1].imshow(mask_fixed ^ mask_registered)
    ax[1, 1].set_title('Difference mask fixed - moving after RR')