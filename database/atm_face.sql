-- phpMyAdmin SQL Dump
-- version 2.11.6
-- http://www.phpmyadmin.net
--
-- Host: localhost
-- Generation Time: Dec 06, 2024 at 05:21 PM
-- Server version: 5.0.51
-- PHP Version: 5.2.6

SET SQL_MODE="NO_AUTO_VALUE_ON_ZERO";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8 */;

--
-- Database: `atm_face`
--

-- --------------------------------------------------------

--
-- Table structure for table `admin`
--

CREATE TABLE `admin` (
  `username` varchar(20) NOT NULL,
  `password` varchar(20) NOT NULL,
  `amount` int(11) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `admin`
--

INSERT INTO `admin` (`username`, `password`, `amount`) VALUES
('admin', 'admin', 12235);

-- --------------------------------------------------------

--
-- Table structure for table `event`
--

CREATE TABLE `event` (
  `id` int(11) NOT NULL,
  `name` varchar(50) NOT NULL,
  `accno` int(11) NOT NULL,
  `amount` int(11) NOT NULL,
  `rdate` varchar(50) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `event`
--

INSERT INTO `event` (`id`, `name`, `accno`, `amount`, `rdate`) VALUES
(1, 'Santhosh', 2147483647, 500, '06-03-2024'),
(2, 'Santhosh', 2147483647, 500, '06-03-2024'),
(3, 'Santhosh', 2147483647, 500, '06-03-2024'),
(4, 'Santhosh', 2147483647, 500, '06-03-2024'),
(5, 'Santhosh', 2147483647, 500, '06-03-2024');

-- --------------------------------------------------------

--
-- Table structure for table `numbers`
--

CREATE TABLE `numbers` (
  `id` int(11) NOT NULL,
  `number` int(11) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `numbers`
--

INSERT INTO `numbers` (`id`, `number`) VALUES
(1, 0),
(2, 1),
(3, 2),
(4, 3),
(5, 4),
(6, 5),
(7, 6),
(8, 7),
(9, 8),
(10, 9);

-- --------------------------------------------------------

--
-- Table structure for table `register`
--

CREATE TABLE `register` (
  `id` int(11) NOT NULL,
  `name` varchar(20) NOT NULL,
  `address` varchar(200) NOT NULL,
  `mobile` bigint(20) NOT NULL,
  `email` varchar(50) NOT NULL,
  `accno` varchar(20) NOT NULL,
  `card` varchar(20) NOT NULL,
  `bank` varchar(20) NOT NULL,
  `branch` varchar(20) NOT NULL,
  `deposit` int(11) NOT NULL,
  `username` varchar(20) NOT NULL,
  `password` varchar(20) NOT NULL,
  `rdate` varchar(20) NOT NULL,
  `aadhar1` varchar(20) NOT NULL,
  `aadhar2` varchar(20) NOT NULL,
  `aadhar3` varchar(20) NOT NULL,
  `face_st` int(11) NOT NULL,
  `fimg` varchar(30) NOT NULL,
  `otp` varchar(20) NOT NULL,
  `allow_st` int(11) NOT NULL,
  `pinno` varchar(20) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `register`
--

INSERT INTO `register` (`id`, `name`, `address`, `mobile`, `email`, `accno`, `card`, `bank`, `branch`, `deposit`, `username`, `password`, `rdate`, `aadhar1`, `aadhar2`, `aadhar3`, `face_st`, `fimg`, `otp`, `allow_st`, `pinno`) VALUES
(1, 'Santhosh', '55, SK Nagar', 9894442716, 'bgeduscanner@gmail.com', '2233440001', '298700017519', 'SBI', 'Chennai', 7500, '', '5680', '06-03-2024', '256385479635', '', '', 0, 'User.1.41.jpg', '', 0, '');

-- --------------------------------------------------------

--
-- Table structure for table `vt_face`
--

CREATE TABLE `vt_face` (
  `id` int(11) NOT NULL,
  `vid` int(11) NOT NULL,
  `vface` varchar(30) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `vt_face`
--

INSERT INTO `vt_face` (`id`, `vid`, `vface`) VALUES
(1, 1, 'User.1.2.jpg'),
(2, 1, 'User.1.3.jpg'),
(3, 1, 'User.1.4.jpg'),
(4, 1, 'User.1.5.jpg'),
(5, 1, 'User.1.6.jpg'),
(6, 1, 'User.1.7.jpg'),
(7, 1, 'User.1.8.jpg'),
(8, 1, 'User.1.9.jpg'),
(9, 1, 'User.1.10.jpg'),
(10, 1, 'User.1.11.jpg'),
(11, 1, 'User.1.12.jpg'),
(12, 1, 'User.1.13.jpg'),
(13, 1, 'User.1.14.jpg'),
(14, 1, 'User.1.15.jpg'),
(15, 1, 'User.1.16.jpg'),
(16, 1, 'User.1.17.jpg'),
(17, 1, 'User.1.18.jpg'),
(18, 1, 'User.1.19.jpg'),
(19, 1, 'User.1.20.jpg'),
(20, 1, 'User.1.21.jpg'),
(21, 1, 'User.1.22.jpg'),
(22, 1, 'User.1.23.jpg'),
(23, 1, 'User.1.24.jpg'),
(24, 1, 'User.1.25.jpg'),
(25, 1, 'User.1.26.jpg'),
(26, 1, 'User.1.27.jpg'),
(27, 1, 'User.1.28.jpg'),
(28, 1, 'User.1.29.jpg'),
(29, 1, 'User.1.30.jpg'),
(30, 1, 'User.1.31.jpg'),
(31, 1, 'User.1.32.jpg'),
(32, 1, 'User.1.33.jpg'),
(33, 1, 'User.1.34.jpg'),
(34, 1, 'User.1.35.jpg'),
(35, 1, 'User.1.36.jpg'),
(36, 1, 'User.1.37.jpg'),
(37, 1, 'User.1.38.jpg'),
(38, 1, 'User.1.39.jpg'),
(39, 1, 'User.1.40.jpg'),
(40, 1, 'User.1.41.jpg');
