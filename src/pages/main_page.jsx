import { VStack, Box, Text, Spinner, useToast, Icon } from "@chakra-ui/react";
import { motion } from "framer-motion";
import useWindowDimensions from "../hooks/window_dimensions";
import ContentSection from "../components/maincontent";
import TagSection from "../components/tagsectionmain";
import { sendInference } from "../API/services/inference_services";
import React, { useState } from "react";
import { FiLink } from "react-icons/fi";

// Создаем анимированные компоненты с помощью motion
const MotionBox = motion(Box);
const MotionVStack = motion(VStack);

const MainPage = () => {
  const { height } = useWindowDimensions();
  const [response, setResponse] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const toast = useToast();

  const fetchTags = async (url) => {
    setIsLoading(true);
    setResponse(null); // Сбрасываем предыдущий результат
    try {
      const data = await sendInference(url);
      console.log("Inference response:", data);
      setResponse(data);
    } catch (err) {
      console.error("Failed to fetch inference data:", err);
      // Показываем ошибку в виде toast-уведомления
      toast({
        title: "Ошибка при загрузке",
        description: "Не удалось получить данные. Проверьте URL видео и попробуйте снова.",
        status: "error",
        duration: 7000,
        isClosable: true,
        position: "top",
      });
    } finally {
      setIsLoading(false);
    }
  };
  
  // Анимационные варианты для framer-motion
  const containerVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { 
      opacity: 1, 
      y: 0,
      transition: { staggerChildren: 0.1, delayChildren: 0.2 }
    },
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0 },
  };

  return (
    <MotionVStack
      minH="100vh"
      width="100%"
      align="center"
      justify="center"
      bg="#ffffff"
      padding={[4, 8, 16]}
      spacing={["16px", "20px", "30px"]}
      initial="hidden"
      animate="visible"
      variants={containerVariants}
    >
      <MotionBox
        width="100%"
        maxW="1000px" // Уменьшаем максимальную ширину блока
        display="flex"
        flexDirection="column"
        alignItems="center"
        bg="rgba(255, 255, 255, 0.6)"
        backdropFilter="blur(10px)"
        border="1px solid rgba(0, 0, 0, 0.1)"
        borderRadius="2xl"
        boxShadow="xl"
        p={[6, 8, 12]}
        variants={itemVariants}
      >
        <ContentSection onFetch={fetchTags} isLoading={isLoading} />
        
        <Box mt={height > 600 ? height * 0.05 : "30px"} minH="150px">
          {isLoading ? (
            <VStack spacing={4}>
              <Spinner
                thickness="4px"
                speed="0.65s"
                emptyColor="gray.200"
                color="blue.500"
                size="xl"
              />
              <Text fontSize="lg" color="gray.600">Анализируем видео...</Text>
            </VStack>
          ) : response ? (
            <MotionBox
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.5 }}
            >
              <TagSection video={response} />
            </MotionBox>
          ) : (
            <VStack spacing={4} color="gray.500" textAlign="center">
              <Icon as={FiLink} boxSize="40px" />
              <Text fontSize="xl" fontWeight="medium">
                Вставьте URL ссылку на видео Rutube
              </Text>
              <Text>Чтобы получить список тегов</Text>
            </VStack>
          )}
        </Box>
      </MotionBox>
    </MotionVStack>
  );
};

export default MainPage;