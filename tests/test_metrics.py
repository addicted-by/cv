import unittest
import cv2
import numpy as np
from skimage.data import chelsea
import sys
sys.path.append('./')
sys.path.append('../')
# Import the functions you want to test
from cv_toolkit.metrics import ssim, psnr
from cv_toolkit.utils.utils import add_noise
from cv_toolkit.utils.processing import load_image
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio
class TestMetrics(unittest.TestCase):

    def setUp(self):
        self.reference_image1 = chelsea()
        self.distorted_image1 = add_noise(self.reference_image1, noise_type="gaussian", intensity=0.1)
        link_to_image = 'https://i.guim.co.uk/img/media/26392d05302e02f7bf4eb143bb84c8097d09144b/446_167_3683_2210/master/3683.jpg?width=1200&quality=85&auto=format&fit=max&s=a52bbe202f57ac0f5ff7f47166906403'

        self.reference_image2, flag = load_image(link_to_image)
        self.reference_image2 = self.reference_image2
        self.assertIsNotNone(flag)
        self.distorted_image2 = add_noise(self.reference_image2, noise_type="gaussian", intensity=0.1)
        self.distorted_image3 = add_noise(self.reference_image2, noise_type="gaussian", intensity=0.2)


        

    def test_ssim(self):
        before_applying_ref = self.reference_image1.copy()
        before_applying_dist = self.distorted_image1.copy()

        ssim_score = ssim(self.reference_image1, self.distorted_image1)
        self.assertTrue(np.allclose(before_applying_ref, self.reference_image1), 
                               msg="Your function modified the image!")
        self.assertTrue(np.allclose(before_applying_dist, self.distorted_image1), 
                               msg="Your function modified the image!")
        

        expected_ssim = structural_similarity(self.reference_image1, 
                                              self.distorted_image1,
                                              channel_axis=-1)
        dummy_ssim = ssim(self.reference_image1, self.reference_image1)
        self.assertAlmostEqual(ssim_score, expected_ssim, delta=0.1)
        self.assertAlmostEqual(dummy_ssim, 1., delta=1e-16)

        ssim_score2 = ssim(self.reference_image2, self.distorted_image2)
        ssim_score3 = ssim(self.reference_image2, self.distorted_image3)
        self.assertLess(ssim_score3, ssim_score2, msg="SSIM does not work well with noise")



    def test_psnr(self):
        before_applying_ref = self.reference_image1.copy()
        before_applying_dist = self.distorted_image1.copy()

        psnr_value = psnr(self.reference_image1, self.distorted_image1)
        self.assertTrue(np.allclose(before_applying_ref, self.reference_image1), 
                               msg="Your function modified the image!")
        self.assertTrue(np.allclose(before_applying_dist, self.distorted_image1), 
                               msg="Your function modified the image!")
        

        expected_psnr = peak_signal_noise_ratio(self.reference_image1, self.distorted_image1)
        dummy_psnr = psnr(self.reference_image1, self.reference_image1)
        self.assertAlmostEqual(psnr_value, expected_psnr, delta=3.0)
        self.assertEqual(dummy_psnr, float('inf'))

        psnr_score2 = psnr(self.reference_image2, self.distorted_image2)
        psnr_score3 = psnr(self.reference_image2, self.distorted_image3)
        self.assertLess(psnr_score3, psnr_score2, msg="PSNR does not work well with noise")


if __name__ == '__main__':
    unittest.main()
